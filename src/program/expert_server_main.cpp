// src/program/expert_server_main.cpp - strata-expert-server: the remote-experts server for a host with NO GPU.
//
// docs/REMOTE_EXPERTS.md, "CPU-only host".  The CUDA engine can serve experts too (`strata ... --expert-server`), but it
// initialises CUDA first, because it loads the dense weights to the card.  A server needs none of that: it answers
// SXP1 REQ frames with the CPU expert kernels over the model's experts, read from the GGUF in place.  This program is
// that and nothing else: it links no CUDA runtime, never calls the driver, and builds with STRATA_ENABLE_CUDA=OFF.
//
// It takes the engine's command line for the server (the same words, so one script serves both builds):
//   strata-expert-server --pack PACK --native SHARD1.gguf --listen 7840 --serve-layers 38-43 [--pool-workers N]
// Flags that only configure the GPU engine (--gpu, --expert-cache, --spec, ...) are accepted and ignored, with one note.
//
// What it shares with generate.cpp's --expert-server path (src/program/generate.cpp, "REMOTE EXPERTS"), on purpose and
// unchanged: the model geometry from the GGUF (core::geometry_from_gguf), the expert layout of the pack, the file expert
// source (FileExpertSource, the GGUF shards mapped in place), the CPU expert pool, `expert_pool_dispatch_multi` (every
// row on the CPU: no plan, no residency table, no cache) and `serve_experts`.  So the rows are the same bits the
// conductor's own CPU pool would have produced.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/net_experts.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <sys/mman.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string pack, native;
    int listen_port = 7840;
    std::string serve_layers;      ///< "a-b" (inclusive); empty: every layer
    int pool_workers = 0;          ///< 0: every physical core minus the host's (the engine's default)
    bool no_host_worker = false;
    bool help = false;
};

void usage(std::FILE* f) {
    std::fprintf(f,
        "strata-expert-server (CPU only: no CUDA is linked or initialised)\n"
        "  --pack DIR           the model's pack folder (native_experts.txt, as for strata --pack)\n"
        "  --native SHARD1      the model's first GGUF shard (the experts are read from the shards in place)\n"
        "  --listen PORT        TCP port (default 7840)\n"
        "  --serve-layers A-B   layers to hold in RAM and answer for, inclusive (default: all)\n"
        "  --pool-workers N     CPU expert pool worker count (default 0: every physical core minus the host's)\n"
        "  --no-host-worker     the calling thread does not take expert jobs (A/B arm)\n"
        "  --expert-server      accepted (this program is the expert server)\n"
        "  ignored, taking the engine's command line: --gpu --max-context --prefill --spec --expert-profile --expert-cache\n"
        "                       --pcie-frac --mmap-experts --ple-gguf --max-new --tokens --greedy\n"
        "Conductor: strata ... --remote-experts HOST:PORT=A-B   (docs/REMOTE_EXPERTS.md)\n");
}

// flags of the GPU engine that a server has no use for: name -> takes a value
bool ignored_flag(const std::string& a, bool& takes_value) {
    static const char* const with_value[] = {"--gpu", "--max-context", "--prefill", "--spec", "--expert-profile", "--expert-cache",
                                             "--pcie-frac", "--ple-gguf", "--max-new", "--tokens"};
    static const char* const bare[] = {"--mmap-experts", "--greedy", "--expert-server"};
    for (const char* f : with_value) if (a == f) { takes_value = true; return true; }
    for (const char* f : bare) if (a == f) { takes_value = false; return true; }
    return false;
}

bool parse(int argc, char** argv, Options& o) {
    bool noted = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-expert-server: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        bool tv = false;
        if (a == "--pack") o.pack = next("--pack");
        else if (a == "--native") o.native = next("--native");
        else if (a == "--listen") o.listen_port = std::atoi(next("--listen"));
        else if (a == "--serve-layers") o.serve_layers = next("--serve-layers");
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--help" || a == "-h") o.help = true;
        else if (ignored_flag(a, tv)) {
            if (tv) (void) next(a.c_str());
            if (!noted && a != "--expert-server") {
                std::fprintf(stderr, "strata-expert-server: GPU-engine flags (%s ...) are ignored: this server has no GPU\n", a.c_str());
                noted = true;
            }
        } else {
            std::fprintf(stderr, "strata-expert-server: unknown option %s (--help)\n", a.c_str());
            return false;
        }
    }
    return true;
}

// Make the served layers' experts resident in RAM: touch every page of every expert blob of the GGUF mapped in place (the
// page cache then holds them) and mlock what `ulimit -l` allows.  The same as the warm-up in generate.cpp's expert server.
struct WarmCtx { strata::core::ExpertSource* src; int64_t ne; };

void warm_served_layers(void* user, int lo, int hi) {
    WarmCtx* w = (WarmCtx*) user;
    if (w->src == nullptr) return;
    const auto t0 = Clock::now();
    uint64_t touched = 0, locked = 0, reads = 0;
    volatile uint8_t sink = 0;
    for (int l = lo; l <= hi; ++l) {
        for (int64_t e = 0; e < w->ne; ++e) {
            strata::core::BlobPart bp[3];
            const int ns = w->src->blob_parts(l, e, bp);
            if (ns == 0) {   // the pread path: a blob read fills the page cache
                const uint8_t* b = w->src->blob(l, e);
                if (b != nullptr) { sink = sink + b[0]; ++reads; }
            }
            for (int i = 0; i < ns; ++i) {
                for (uint64_t off = 0; off < bp[i].n; off += 4096) sink = sink + bp[i].p[off];
                touched += bp[i].n;
                if (mlock(bp[i].p, bp[i].n) == 0) locked += bp[i].n;
            }
        }
    }
    (void) sink;
    const auto* fs = dynamic_cast<const strata::core::FileExpertSource*>(w->src);
    uint64_t have = 0, want = 0;
    for (int l = lo; l <= hi; ++l)
        for (int64_t e = 0; e < w->ne; e += 8) {
            uint64_t total = 0;
            const uint64_t c = fs != nullptr ? fs->resident_bytes(l, e, total) : 0;
            have += c; want += total;
        }
    std::fprintf(stderr, "strata net: served layers %d-%d: touched %.2f GiB mapped + %llu blob reads, mlocked %.2f GiB, "
                         "page-cache resident %.2f of %.2f GiB (sampled 1 in 8) in %.1f s\n",
                 lo, hi, (double) touched / 1073741824.0, (unsigned long long) reads, (double) locked / 1073741824.0,
                 (double) have * 8.0 / 1073741824.0, (double) want * 8.0 / 1073741824.0,
                 std::chrono::duration<double>(Clock::now() - t0).count());
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) return 2;
    if (o.help) { usage(stdout); return 0; }
    if (o.pack.empty() || o.native.empty()) {
        std::fprintf(stderr, "strata-expert-server: --pack and --native are required (--help)\n");
        return 2;
    }
    if (o.listen_port < 1 || o.listen_port > 65535 || o.pool_workers < 0) {
        std::fprintf(stderr, "strata-expert-server: --listen is 1..65535 and --pool-workers >= 0\n");
        return 2;
    }
    std::string err;
    // The CPU kernels the pack needs: a native (i-quant) pack runs on AVX2 (ggml-cpu below that); only the canonical Q2_0
    // pack needs AVX-512 (generate.cpp does the same check).
    if (!strata::kernels::cpu::cpu_avx2_ok()) {
        std::fprintf(stderr, "strata-expert-server: this CPU (%s) has no AVX2 with FMA and F16C, which the CPU expert kernels need\n",
                     strata::kernels::cpu::cpu_name().c_str());
        return 2;
    }
    // The model's geometry from the first shard, as generate.cpp reads it for --native.
    strata::core::ModelGeometry g;
    int64_t K = g.n_expert_used;
    std::vector<std::string> shards;
    try {
        shards = strata::gguf_split_paths(o.native);
        strata::GgufFile meta(shards.front());
        if (!strata::core::geometry_from_gguf(meta, g, K, err)) {
            std::fprintf(stderr, "strata-expert-server: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-expert-server: reading the model's geometry from %s: %s\n", o.native.c_str(), e.what());
        return 1;
    }
    if (!strata::kernels::cpu::expert_layout_load(o.pack, g.n_layers, g.n_expert, err)) {
        std::fprintf(stderr, "strata-expert-server: %s\n", err.c_str());
        return 1;
    }
    if (!strata::kernels::cpu::expert_layout().native && !strata::kernels::cpu::cpu_features().usable()) {
        std::fprintf(stderr, "strata-expert-server: %s has no native_experts.txt, so it is the canonical Q2_0 pack, whose CPU "
                             "kernel needs AVX-512 (%s has none); use a native (i-quant) pack\n", o.pack.c_str(),
                     strata::kernels::cpu::cpu_name().c_str());
        return 2;
    }
    std::fprintf(stderr, "strata-expert-server: CPU only, CUDA not linked. model: %lld layers, %lld experts (top-%lld), n_embd %lld; %s\n",
                 (long long) g.n_layers, (long long) g.n_expert, (long long) K, (long long) g.n_embd,
                 strata::kernels::cpu::cpu_name().c_str());

    // The experts: the GGUF shards mapped in place, no copy (FileExpertSource, the engine's --mmap-experts).
    strata::core::FileExpertSource src;
    src.set_gguf(o.native);
    if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
        std::fprintf(stderr, "strata-expert-server: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-expert-server: experts via mmap (%s)\n",
                 src.gguf_mode() ? "the GGUF shards in place, no experts.bin" : "experts.bin");

    strata::kernels::cpu::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker,
                                          strata::kernels::cpu::PoolAffinity::All, /*tasks=*/0);
    std::fprintf(stderr, "strata-expert-server: CPU pool: %d workers + %s calling thread\n", pool.workers(),
                 pool.host_works() ? "the" : "no");

    strata::core::ExpertDispatch d;
    d.pool = &pool;
    d.src = &src;
    d.n_expert = g.n_expert;
    d.jobs.resize((size_t) K);

    strata::core::ExpertServeCfg sc;
    sc.port = o.listen_port;
    sc.n_layers = g.n_layers;
    sc.n_expert = g.n_expert;
    sc.layer_lo = 0;
    sc.layer_hi = (int) g.n_layers - 1;
    if (!o.serve_layers.empty()) {
        const size_t dash = o.serve_layers.find('-');
        sc.layer_lo = std::atoi(o.serve_layers.c_str());
        sc.layer_hi = dash == std::string::npos ? sc.layer_lo : std::atoi(o.serve_layers.c_str() + dash + 1);
    }
    if (sc.layer_lo < 0 || sc.layer_hi < sc.layer_lo || sc.layer_hi >= g.n_layers) {
        std::fprintf(stderr, "strata-expert-server: --serve-layers a-b is outside 0..%lld\n", (long long) g.n_layers - 1);
        return 2;
    }
    WarmCtx wc{&src, g.n_expert};
    sc.warm_user = &wc;
    sc.warm = &warm_served_layers;
    return strata::core::serve_experts(d, sc);
}
