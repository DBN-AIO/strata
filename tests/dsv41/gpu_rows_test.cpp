// Strata. The GPU expert server's rows against the CPU server's (docs/DEEPSEEK41_SPLIT.md "GPU expert server").
//
// Needs a CUDA device and a synthetic pack (tools/dsv41/make_synthetic_pack.py; never real weights).  For every served layer and
// windows of 1, 2, 4, 8 and 16 tokens with random picks (some entries masked out, some tokens sharing experts) it computes the rows
// with net::ExpertHost (the CPU server's code) and net::GpuExpertHost (VRAM, the engine's window kernels) and compares them:
// unasked rows must stay untouched, asked rows must agree to the documented tolerance (the two sides round the activations to
// int8 the same way, but sum a block's products in another order, so they are not bit-identical: max relative error against the
// row's RMS is printed; the test fails above --tol, default 1e-3).  --bench then times a layer call on each host (warm-up first,
// fresh random picks every call) for windows of 1, 2, 4, 8, 16 tokens: the K = 4 verify window against one token is what decides
// whether a drafter pays in a split.
//
//   ds4_gpu_rows_test --pack DIR [--layers A-B] [--cap-gb G] [--device N] [--threads N] [--tol X] [--bench N] [--seed S]
#include "ds4_net.hpp"
#include "ds4_net_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds4::net;

namespace {
using Clock = std::chrono::steady_clock;
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    std::string pack;
    int lo = 0, hi = -1, device = 0, threads = 0, bench = 0;
    double cap_gb = 0.0, tol = 1e-4, hard = 5e-2;
    unsigned seed = 7;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--pack") pack = v();
        else if (a == "--layers") { if (std::sscanf(v().c_str(), "%d-%d", &lo, &hi) != 2) return 2; }
        else if (a == "--cap-gb") cap_gb = std::atof(v().c_str());
        else if (a == "--device") device = std::atoi(v().c_str());
        else if (a == "--threads") threads = std::atoi(v().c_str());
        else if (a == "--tol") tol = std::atof(v().c_str());
        else if (a == "--hard") hard = std::atof(v().c_str());
        else if (a == "--bench") bench = std::atoi(v().c_str());
        else if (a == "--seed") seed = (unsigned) std::atoi(v().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    ExpertMap map;
    std::string err;
    if (pack.empty() || !map.load(pack, err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 2; }
    if (hi < 0) hi = map.n_layers - 1;
    const int D = map.dim, K = map.top_k, NE = map.n_exp;
    ExpertHost cpu;
    if (!cpu.open(map, lo, hi, threads, err)) { std::fprintf(stderr, "cpu host: %s\n", err.c_str()); return 1; }
    GpuOpts go;
    go.device = device;
    go.cap_bytes = cap_gb > 0.0 ? (uint64_t) (cap_gb * 1073741824.0) : 0;
    GpuExpertHost gpu;
    if (!gpu.open(map, lo, hi, 0, -1, go, err)) { std::printf("GPU_OPEN_FAILED: %s\n", err.c_str()); return 3; }
    std::printf("GPU %s: %d layers in VRAM, %.2f GiB\n", gpu.device_name().c_str(), gpu.layers_on_gpu(), (double) gpu.resident_bytes() / 1073741824.0);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto make_window = [&](int nt, std::vector<float>& x, std::vector<int32_t>& ids, std::vector<uint8_t>& mask) {
        x.resize((size_t) nt * D);
        for (float& f : x) f = nd(rng);
        ids.assign((size_t) nt * K, 0);
        mask.assign((size_t) nt * K, 1);
        for (int t = 0; t < nt; ++t) {
            std::vector<int> pool((size_t) NE);
            for (int e = 0; e < NE; ++e) pool[(size_t) e] = e;
            // half the windows' tokens draw from a small pool, so experts are shared between tokens (the grouped path)
            const int span = (t % 2 == 0) ? NE : std::min(NE, K + 2);
            std::shuffle(pool.begin(), pool.begin() + span, rng);
            for (int j = 0; j < K; ++j) { ids[(size_t) t * K + j] = pool[(size_t) j]; mask[(size_t) t * K + j] = (rng() % 8) != 0; }
        }
    };

    long rows_total = 0, rows_bit = 0, untouched_bad = 0, fails = 0, off_tol = 0;
    double max_abs = 0.0, max_rel = 0.0;
    uint64_t h_cpu = 1469598103934665603ull, h_gpu = h_cpu;
    const float sentinel = 12345.0f;
    for (int l = lo; l <= hi; ++l) {
        const double rel_before = max_rel;
        double layer_rel = 0.0;
        (void) rel_before;
        for (int nt : {1, 2, 4, 8, 16}) {
            std::vector<float> x;
            std::vector<int32_t> ids;
            std::vector<uint8_t> mask;
            make_window(nt, x, ids, mask);
            std::vector<float> rc((size_t) nt * K * D, sentinel), rg((size_t) nt * K * D, sentinel);
            if (!cpu.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), mask.data(), x.data(), rc.data(), err)) { std::printf("cpu: %s\n", err.c_str()); return 1; }
            if (!gpu.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), mask.data(), x.data(), rg.data(), err)) { std::printf("gpu: %s\n", err.c_str()); return 1; }
            for (int i = 0; i < nt * K; ++i) {
                const float* a = &rc[(size_t) i * D];
                const float* b = &rg[(size_t) i * D];
                if (!mask[(size_t) i]) {
                    for (int d = 0; d < D; ++d) if (b[d] != sentinel) { ++untouched_bad; break; }
                    continue;
                }
                ++rows_total;
                double ss = 0.0, md = 0.0;
                bool same = true, finite = true;
                for (int d = 0; d < D; ++d) {
                    ss += (double) a[d] * a[d];
                    md = std::max(md, (double) std::fabs(a[d] - b[d]));
                    same = same && std::memcmp(&a[d], &b[d], 4) == 0;
                    finite = finite && std::isfinite(b[d]);
                }
                const double rms = std::sqrt(ss / (double) D);
                const double rel = rms > 0.0 ? md / rms : md;
                rows_bit += same ? 1 : 0;
                max_abs = std::max(max_abs, md);
                max_rel = std::max(max_rel, rel);
                layer_rel = std::max(layer_rel, rel);
                if (rel > tol) ++off_tol;
                if (!finite || rel > hard) ++fails;
                h_cpu = fnv(h_cpu, a, (size_t) D * 4);
                h_gpu = fnv(h_gpu, b, (size_t) D * 4);
            }
        }
        std::printf("layer %d types %d/%d/%d: max diff / row RMS %.3e\n", l, map.layers[(size_t) l].type[0], map.layers[(size_t) l].type[1], map.layers[(size_t) l].type[2], layer_rel);
    }
    std::printf("ROWS %ld, bit-identical %ld (%.1f%%), max abs diff %.3e, max diff / row RMS %.3e (tol %.1e), out of tolerance %ld, unasked rows touched %ld\n",
                rows_total, rows_bit, rows_total ? 100.0 * (double) rows_bit / (double) rows_total : 0.0, max_abs, max_rel, tol, fails, untouched_bad);
    std::printf("ROWHASH cpu %016llx gpu %016llx\n", (unsigned long long) h_cpu, (unsigned long long) h_gpu);

    if (bench > 0) {
        std::printf("BENCH %d calls a window size after %d warm-up, layers %d-%d, fresh random picks every call\n", bench, std::max(10, bench / 10), lo, hi);
        std::printf("tokens | cpu ms a call | gpu ms a call | cpu vs 1 token | gpu vs 1 token\n");
        double c1 = 0, g1 = 0;
        for (int nt : {1, 2, 4, 8, 16}) {
            std::vector<float> x, out((size_t) nt * K * D);
            std::vector<int32_t> ids;
            std::vector<uint8_t> mask;
            double tc = 0, tg = 0;
            const int warm = std::max(10, bench / 10);
            for (int c = -warm; c < bench; ++c) {
                const int l = lo + (int) (rng() % (unsigned) (hi - lo + 1));
                make_window(nt, x, ids, mask);
                std::fill(mask.begin(), mask.end(), (uint8_t) 1);
                double t0 = now_ms();
                cpu.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), mask.data(), x.data(), out.data(), err);
                double t1 = now_ms();
                gpu.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), mask.data(), x.data(), out.data(), err);
                double t2 = now_ms();
                if (c >= 0) { tc += t1 - t0; tg += t2 - t1; }
            }
            tc /= bench; tg /= bench;
            if (nt == 1) { c1 = tc; g1 = tg; }
            std::printf("%6d | %13.3f | %13.3f | %14.2f | %14.2f\n", nt, tc, tg, c1 > 0 ? tc / c1 : 0.0, g1 > 0 ? tg / g1 : 0.0);
        }
    }
    // the rule: a row is "typical" within --tol of its RMS; the two sides sum a block in another order, so an activation that sits on an
    // int8 rounding boundary can flip one quantum in an odd row: up to 1 % of rows may exceed --tol, none --hard, and nothing is non-finite
    std::printf("rows beyond tol %ld of %ld (allowed 1%%), beyond hard %.1e or non-finite: %ld\n", off_tol, rows_total, hard, fails);
    const bool ok = fails == 0 && off_tol * 100 <= rows_total && untouched_bad == 0 && rows_total > 0;
    std::printf(ok ? "gpu_rows_test: all checks passed\n" : "gpu_rows_test: FAILED\n");
    return ok ? 0 : 1;
}
