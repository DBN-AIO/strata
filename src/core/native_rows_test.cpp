// src/core/native_rows_test.cpp - the qwen35moe family's CPU rows and the expert split, on a Max-shaped synthetic GGUF (CPU only).
//
//   native_rows_test MODEL.gguf     (tools/make_synthetic_max.py writes one; tests/qwen_max/run_cpu_test.py does both)
//
// 1. window invariance: with arith 1 a row does not depend on the window it was computed in (token alone == token among eight);
// 2. the split: an expert server (serve_experts over NativeRows) on loopback and a NetExperts client (the conductor's prompt path,
//    windows of kWindowMaxT tokens) return, for every masked entry, exactly the bytes NativeRows gives in process;
// 3. an identity of another model file is refused in the handshake (the fingerprint covers the expert tensors' types and shapes).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/native_rows.hpp"
#include "strata/core/net_experts.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/window_limits.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

namespace C = strata::kernels::cpu;
using strata::core::NativeLayerBlobs;
using strata::core::NativeRows;

static int g_fail = 0;
#define CHECK(c, ...)                                                                                      \
    do {                                                                                                   \
        if (!(c)) {                                                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #c);                              \
            std::fprintf(stderr, __VA_ARGS__);                                                             \
            std::fprintf(stderr, "\n");                                                                    \
            ++g_fail;                                                                                      \
        }                                                                                                  \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: native_rows_test MODEL.gguf\n"); return 2; }
    const std::string path = argv[1];
    strata::GgufModel gm = strata::GgufModel::open(path);
    strata::core::ModelGeometry g;
    int64_t K = 0;
    std::string err;
    if (!strata::core::geometry_from_gguf(gm.meta(), g, K, err) || !strata::core::check_gguf_layout(gm, g, err)) {
        std::fprintf(stderr, "model: %s\n", err.c_str());
        return 1;
    }
    const int total = (int) (g.n_layers + g.n_mtp_layers);
    std::vector<int> picks = {0, 1};   // two trunk blocks and, when the file has one, the MTP block (Q2_K experts in the Max files)
    if (g.n_mtp_layers > 0) picks.push_back((int) g.n_layers);
    std::printf("native_rows_test: %lld blocks + %lld MTP, %lld experts top-%lld, n_embd %lld n_ff %lld; testing %zu blocks\n", (long long) g.n_layers,
                (long long) g.n_mtp_layers, (long long) g.n_expert, (long long) K, (long long) g.n_embd, (long long) g.n_ff, picks.size());

    std::vector<std::vector<uint8_t>> mem(picks.size());
    std::vector<NativeLayerBlobs> layers((size_t) total);
    for (size_t pi = 0; pi < picks.size(); ++pi) {
        const int l = picks[pi];
        NativeLayerBlobs& L = layers[(size_t) l];
        const strata::TensorInfo* tg = gm.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const strata::TensorInfo* td = gm.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        CHECK(C::native_fmt((int) tg->type, (int) td->type, g.n_embd, g.n_ff, L.fmt, err), "block %d: %s", l, err.c_str());
        L.stride = (L.fmt.bytes + 63) / 64 * 64;
        mem[pi].assign(L.stride * (size_t) g.n_expert + 64, 0);
        uint8_t* base = mem[pi].data() + ((64 - (reinterpret_cast<uintptr_t>(mem[pi].data()) & 63)) & 63);
        for (int64_t e = 0; e < g.n_expert; ++e)
            CHECK(strata::core::native_expert_blob(gm, l, e, L.fmt, base + (size_t) e * L.stride, err), "%s", err.c_str());
        L.base = base;
    }
    if (g_fail) return 1;
    C::ExpertPool pool(4, /*pin=*/false, /*host_works=*/true, C::PoolAffinity::All, 0);
    const size_t N = (size_t) g.n_embd;
    NativeRows rows(layers, g.n_embd, g.n_expert, pool);

    // a request: T tokens, k experts each, a third of the entries not asked for
    std::mt19937 rng(7);
    const int T = 21, k = (int) K;   // 21 tokens = three windows of kWindowMaxT = 8 (8 + 8 + 5)
    std::vector<float> x((size_t) T * N);
    for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) * 0.002f;
    std::vector<int32_t> ids((size_t) T * k);
    std::vector<uint8_t> mask((size_t) T * k);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) {
            int e;
            bool dup;
            do {
                e = (int) (rng() % (uint32_t) g.n_expert);
                dup = false;
                for (int q = 0; q < j; ++q) dup = dup || ids[(size_t) t * k + q] == e;
            } while (dup);
            ids[(size_t) t * k + j] = e;
            mask[(size_t) t * k + j] = rng() % 3 != 0;
        }
    const size_t n = (size_t) T * k;

    // 1. window invariance
    std::vector<float> whole(n * N, 0.0f), alone(n * N, 0.0f);
    CHECK(rows.run(0, T, k, ids.data(), mask.data(), x.data(), whole.data(), C::NativeArithmetic::WindowInvariant, err), "%s", err.c_str());
    for (int t = 0; t < T; ++t)
        CHECK(rows.run(0, 1, k, ids.data() + (size_t) t * k, mask.data() + (size_t) t * k, x.data() + (size_t) t * N, alone.data() + (size_t) t * k * N,
                       C::NativeArithmetic::WindowInvariant, err), "%s", err.c_str());
    size_t asked = 0, differ = 0;
    for (size_t i = 0; i < n; ++i)
        if (mask[i]) {
            ++asked;
            if (std::memcmp(&whole[i * N], &alone[i * N], N * 4) != 0) ++differ;
        }
    CHECK(differ == 0, "%zu of %zu rows depend on their window under the window-invariant arithmetic", differ, asked);
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < n; ++i)
        if (mask[i])
            for (size_t c = 0; c < N; ++c) { finite = finite && std::isfinite(whole[i * N + c]); nonzero = nonzero || whole[i * N + c] != 0.0f; }
    CHECK(finite && nonzero, "rows are not finite and non-zero (finite %d nonzero %d)", (int) finite, (int) nonzero);
    std::printf("  window invariance: %zu rows, %zu differ\n", asked, differ);

    // 2. the split over loopback
    const strata::core::NetIdentity id = strata::core::native_net_identity(gm, g, K, {path});
    const int port = 21000 + (int) (getpid() % 20000);
    strata::core::ExpertServeCfg sc;
    sc.port = port;
    sc.n_layers = total;
    sc.n_expert = g.n_expert;
    sc.top_k = K;
    sc.identity = id;
    sc.layer_lo = 0;
    sc.layer_hi = total - 1;
    sc.once = true;
    NativeRows server_rows(layers, g.n_embd, g.n_expert, pool);
    const strata::core::ExpertComputeFn compute = [&](uint32_t layer, uint32_t nt, uint32_t kk, int arith, const int32_t* i2, const uint8_t* m2, const float* x2,
                                                   float* out, std::string& e) {
        return server_rows.run((int) layer, (int) nt, (int) kk, i2, m2, x2, out, arith == 1 ? C::NativeArithmetic::WindowInvariant : C::NativeArithmetic::Default, e);
    };
    int serve_rc = -1;
    std::thread server([&] { serve_rc = strata::core::serve_experts(sc, compute); });
    {
        // the right conductor first (the server may still be starting); then one with another model's identity is refused
        strata::core::NetExperts net;
        std::string e3;
        bool up = false;
        for (int tries = 0; tries < 50 && !up; ++tries) {
            up = net.open("127.0.0.1:" + std::to_string(port) + "=0-" + std::to_string(total - 1), total, g.n_expert, id, e3);
            if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        CHECK(up, "open: %s", e3.c_str());
        strata::core::NetIdentity wrong = id;
        wrong.fingerprint[0] ^= 1;
        strata::core::NetExperts bad;
        std::string e2;
        const bool opened = bad.open("127.0.0.1:" + std::to_string(port) + "=0-" + std::to_string(total - 1), total, g.n_expert, wrong, e2);
        CHECK(!opened && e2.find("fingerprint") != std::string::npos, "a wrong fingerprint was not refused (%s)", e2.c_str());
        if (up) {
            for (const int l : picks) {
                std::vector<float> got(n * N, 0.0f), ref(n * N, 0.0f);
                std::vector<float*> dst(n, nullptr);
                for (size_t i = 0; i < n; ++i) dst[i] = &got[i * N];
                std::string e4;
                CHECK(net.prompt(l, x.data(), ids.data(), T, k, mask.data(), dst.data(), e4), "prompt block %d: %s", l, e4.c_str());
                CHECK(rows.run(l, T, k, ids.data(), mask.data(), x.data(), ref.data(), C::NativeArithmetic::WindowInvariant, err), "%s", err.c_str());
                size_t bad_rows = 0;
                for (size_t i = 0; i < n; ++i)
                    if (mask[i] && std::memcmp(&got[i * N], &ref[i * N], N * 4) != 0) ++bad_rows;
                CHECK(bad_rows == 0, "block %d: %zu of %zu rows from the server differ from the in-process rows", l, bad_rows, asked);
                std::printf("  split, block %d: %zu rows through the server, %zu differ\n", l, asked, bad_rows);
            }
        }
    }   // the client's destructor says BYE: the one-connection server returns
    server.join();
    CHECK(serve_rc == 0, "serve_experts returned %d", serve_rc);
    std::printf(g_fail ? "native_rows_test: FAILED (%d)\n" : "native_rows_test: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
