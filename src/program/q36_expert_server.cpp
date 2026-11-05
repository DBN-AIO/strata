// src/program/q36_expert_server.cpp - the qwen35moe family's side of strata-expert-server (CPU only, no CUDA).
//
// `strata-expert-server --native SHARD1.gguf --listen HOST:PORT --serve-layers LO-HI` on a qwen35moe model (Qwen3.6-35B-A3B, Qwen3.8-Max
// 2.4T-A95B): no --pack, the experts are repacked from the GGUF into RAM blobs [gate | up | down] - the engine's own native blob
// (src/qwen35/q36_engine.cu, the hybrid mode's host copy) - and every REQ is answered with `NativeRows`, the same CPU kernels and
// job grouping the engine's pool uses, so the rows are the bits the conductor's own CPU would have produced (docs/QWEN_MAX.md).
// Blocks 0..n_layers-1 are the trunk; block n_layers is the in-file MTP block, which a conductor may serve too.
#include "q36_expert_server.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/native_rows.hpp"
#include "strata/core/net_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata_q36 {

namespace C = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

int serve(const Options& o, const strata::core::ModelGeometry& g, int64_t top_k) {
    std::string err;
    std::vector<std::string> files;
    try {
        files = strata::gguf_split_paths(o.native);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-expert-server: %s\n", e.what());
        return 1;
    }
    std::unique_ptr<strata::GgufModel> gmp;
    try {
        gmp = std::make_unique<strata::GgufModel>(files);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-expert-server: %s\n", e.what());
        return 1;
    }
    const strata::GgufModel& gm = *gmp;
    if (!strata::core::check_gguf_layout(gm, g, err)) {
        std::fprintf(stderr, "strata-expert-server: %s\n", err.c_str());
        return 1;
    }
    const int total = (int) (g.n_layers + g.n_mtp_layers);
    int lo = 0, hi = total - 1;
    if (!o.serve_layers.empty()) {
        const size_t dash = o.serve_layers.find('-');
        lo = std::atoi(o.serve_layers.c_str());
        hi = dash == std::string::npos ? lo : std::atoi(o.serve_layers.c_str() + dash + 1);
    }
    if (lo < 0 || hi < lo || hi >= total) {
        std::fprintf(stderr, "strata-expert-server: --serve-layers a-b is outside 0..%d (the trunk is 0..%lld, the MTP block %lld)\n", total - 1,
                     (long long) g.n_layers - 1, (long long) g.n_layers);
        return 2;
    }
    std::fprintf(stderr, "strata-expert-server: CPU only, CUDA not linked. %s model: %lld blocks + %lld MTP, %lld experts (top-%lld), n_embd %lld, n_ff %lld; %s\n",
                 strata::model_family_title(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers, (long long) g.n_expert, (long long) top_k,
                 (long long) g.n_embd, (long long) g.n_ff, C::cpu_name().c_str());

    // the blocks' formats, then their experts in RAM (an anonymous mapping, huge pages where the kernel offers them)
    std::vector<strata::core::NativeLayerBlobs> layers((size_t) total);
    std::vector<size_t> bytes((size_t) total, 0);
    size_t all = 0;
    for (int l = lo; l <= hi; ++l) {
        const strata::TensorInfo* tg = gm.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const strata::TensorInfo* td = gm.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        strata::core::NativeLayerBlobs& L = layers[(size_t) l];
        if (!C::native_fmt((int) tg->type, (int) td->type, g.n_embd, g.n_ff, L.fmt, err)) {
            std::fprintf(stderr, "strata-expert-server: block %d: %s\n", l, err.c_str());
            return 1;
        }
        L.stride = (L.fmt.bytes + 63) / 64 * 64;
        bytes[(size_t) l] = L.stride * (size_t) g.n_expert;
        all += bytes[(size_t) l];
    }
    std::fprintf(stderr, "strata-expert-server: serving blocks %d-%d: %.2f GiB of experts to repack into RAM\n", lo, hi, (double) all / 1073741824.0);
    const auto t0 = Clock::now();
    std::atomic<int> next{lo};
    std::atomic<bool> failed{false};
    std::string first_err;
    std::mutex em;
    const int nthreads = std::max(1, std::min<int>((int) std::thread::hardware_concurrency() / 2, hi - lo + 1));
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&] {
            for (;;) {
                const int l = next.fetch_add(1);
                if (l > hi || failed.load()) return;
                strata::core::NativeLayerBlobs& L = layers[(size_t) l];
                void* p = mmap(nullptr, bytes[(size_t) l], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    std::lock_guard<std::mutex> lk(em);
                    first_err = "no memory for block " + std::to_string(l) + " (" + std::to_string(bytes[(size_t) l] >> 20) + " MiB)";
                    failed = true;
                    return;
                }
#ifdef MADV_HUGEPAGE
                madvise(p, bytes[(size_t) l], MADV_HUGEPAGE);
#endif
                uint8_t* base = (uint8_t*) p;
                std::string e2;
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (!strata::core::native_expert_blob(gm, l, e, L.fmt, base + (size_t) e * L.stride, e2)) {
                        std::lock_guard<std::mutex> lk(em);
                        first_err = e2;
                        failed = true;
                        return;
                    }
                (void) mlock(base, bytes[(size_t) l]);   // best effort; ulimit -l decides
                L.base = base;
            }
        });
    for (auto& t : th) t.join();
    if (failed.load()) {
        std::fprintf(stderr, "strata-expert-server: %s\n", first_err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-expert-server: %.2f GiB resident in %.1f s\n", (double) all / 1073741824.0,
                 std::chrono::duration<double>(Clock::now() - t0).count());

    C::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker, C::PoolAffinity::All, /*tasks=*/0);
    std::fprintf(stderr, "strata-expert-server: CPU pool: %d workers + %s calling thread\n", pool.workers(), pool.host_works() ? "the" : "no");
    strata::core::NativeRows rows(std::move(layers), g.n_embd, g.n_expert, pool);

    strata::core::ExpertServeCfg sc;
    sc.port = o.listen_port;
    if (!o.listen_host.empty()) sc.host = o.listen_host;
    sc.n_layers = total;
    sc.n_expert = g.n_expert;
    sc.top_k = top_k;
    sc.identity = strata::core::native_net_identity(gm, g, top_k, files);
    sc.layer_lo = lo;
    sc.layer_hi = hi;
    const strata::core::ExpertComputeFn compute = [&rows](uint32_t layer, uint32_t n_tok, uint32_t k, int arith, const int32_t* ids, const uint8_t* mask,
                                                       const float* x, float* out, std::string& e) {
        return rows.run((int) layer, (int) n_tok, (int) k, ids, mask, x, out,
                        arith == 1 ? C::NativeArithmetic::WindowInvariant : C::NativeArithmetic::Default, e);
    };
    return strata::core::serve_experts(sc, compute);
}

}  // namespace strata_q36
