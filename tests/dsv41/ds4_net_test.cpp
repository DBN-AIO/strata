// Strata. The synthetic split test's program for DeepSeek experts over the LAN (tools/dsv41/synthetic_split_test.py).
//
// A tiny DeepSeek-V4.1-shaped model: layers of routed experts (n_exp, top-k, gate / up / down in the quant types the real
// sets use, from a pack that tools/dsv41/make_synthetic_pack.py wrote), a router, a shared expert and a residual, driven
// greedily for N tokens from a prompt that goes through as one window.  The experts are computed
//   --mode local   in this process (net::ExpertHost: the code the server runs)
//   --mode remote  by expert servers (net::Remote, the conductor's side), with this process's own host as the fallback
// and the program prints the generated tokens and a hash of every expert row it received (bit identity, not just argmax).
// Equal lines from the two modes, with a server on localhost or on another PC, is the test.
#include "ds4_net.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds4::net;

namespace {
uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
void rms_norm(const float* x, float* y, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float s = 1.0f / std::sqrt(ss / (float) n + 1e-6f);
    for (int i = 0; i < n; ++i) y[i] = x[i] * s;
}
}  // namespace

int main(int argc, char** argv) {
    std::string pack, mode = "local", remote_spec;
    int n_new = 64, n_prompt = 6, threads = 2, vocab = 64;
    unsigned seed = 1;
    long arch = -1;
    bool expect_open_fail = false;
    // BBXL S3: --skew S makes the router prefer a hidden per-layer popularity order (score += S * -ln(1 + hidden rank)), so a real-shape toy has
    // hot and cold experts; --count-out FILE writes the picks as the engine's expert profile (uint32 [layer][expert], what plan_split.py
    // --popularity reads); --time prints the decode's ms a token.  All three are off by default and change nothing then.
    float skew = 0.0f;
    std::string count_out;
    bool show_time = false;
    // BBXL S4: --overlap asks the servers with Remote::begin, computes the conductor's own experts while they answer, then Remote::finish
    // (the engine's cpu_experts); off: Remote::run, then the local ones.  --local-delay-ms N charges every layer N ms of conductor-side lane
    // time (a sleep: deterministic on a loaded host), so the timing tests can tell "the sum" from "the longer of the two".
    bool overlap = false;
    int salt = 0;   // --prompt-salt N: another prompt (a unique prefix for every timing run)
    int local_delay_ms = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--pack") pack = v();
        else if (a == "--mode") mode = v();
        else if (a == "--remote") remote_spec = v();
        else if (a == "--tokens") n_new = std::atoi(v().c_str());
        else if (a == "--prompt") n_prompt = std::atoi(v().c_str());
        else if (a == "--threads") threads = std::atoi(v().c_str());
        else if (a == "--seed") seed = (unsigned) std::atoi(v().c_str());
        else if (a == "--arch") arch = std::atol(v().c_str());
        else if (a == "--expect-open-fail") expect_open_fail = true;
        else if (a == "--skew") skew = (float) std::atof(v().c_str());
        else if (a == "--count-out") count_out = v();
        else if (a == "--time") show_time = true;
        else if (a == "--overlap") overlap = true;
        else if (a == "--prompt-salt") salt = std::atoi(v().c_str());
        else if (a == "--local-delay-ms") local_delay_ms = std::atoi(v().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    ExpertMap map;
    std::string err;
    if (pack.empty() || !map.load(pack, err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 2; }
    const int D = map.dim, L = map.n_layers, K = map.top_k, NE = map.n_exp;
    if (n_prompt < 1 || n_prompt > kMaxTok) { std::fprintf(stderr, "--prompt 1..%d\n", kMaxTok); return 2; }

    ExpertHost host;
    if (!host.open(map, 0, L - 1, threads, err)) { std::fprintf(stderr, "host: %s\n", err.c_str()); return 1; }
    Remote remote;
    if (mode == "remote") {
        if (!remote.open(map, remote_spec, err, arch < 0 ? 0xffffffffu : (uint32_t) arch)) {
            std::printf("OPEN_FAILED: %s\n", err.c_str());
            return expect_open_fail ? 0 : 1;
        }
        if (expect_open_fail) { std::printf("OPEN_UNEXPECTEDLY_OK\n"); return 1; }
    } else if (mode != "local") { std::fprintf(stderr, "--mode local|remote\n"); return 2; }

    // the toy model's dense part: the same seed gives the same weights in every process
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](std::vector<float>& v, size_t n, float s) { v.resize(n); for (float& f : v) f = nd(rng) * s; };
    std::vector<float> emb, gate, sh1, sh2, outw;
    const int SH = 32;
    fill(emb, (size_t) vocab * D, 1.0f);
    fill(gate, (size_t) L * NE * D, 1.0f / std::sqrt((float) D));
    fill(sh1, (size_t) L * SH * D, 1.0f / std::sqrt((float) D));
    fill(sh2, (size_t) L * D * SH, 1.0f / std::sqrt((float) SH));
    fill(outw, (size_t) vocab * D, 1.0f / std::sqrt((float) D));

    std::vector<float> hot_bias;                 // [layer][expert]: -skew * ln(1 + hidden rank), the hidden order seeded per layer
    std::vector<uint32_t> counts;
    if (skew > 0.0f) {
        hot_bias.assign((size_t) L * NE, 0.0f);
        for (int l = 0; l < L; ++l) {
            std::vector<int> o((size_t) NE);
            for (int e = 0; e < NE; ++e) o[(size_t) e] = e;
            std::mt19937 pr(7919u + (unsigned) l);
            std::shuffle(o.begin(), o.end(), pr);
            for (int r = 0; r < NE; ++r) hot_bias[(size_t) l * NE + (size_t) o[(size_t) r]] = -skew * std::log(1.0f + (float) r);
        }
    }
    if (!count_out.empty()) counts.assign((size_t) L * NE, 0u);
    uint64_t row_hash = 14695981039346656037ull, flag = 0;
    long rows_total = 0;
    std::vector<int> seq;
    std::vector<int> prompt;
    for (int i = 0; i < n_prompt; ++i) prompt.push_back((int) ((i * 7 + 3 + salt * 13) % vocab));

    // one window of nt tokens (their hidden states h: nt x D) through every layer
    auto forward = [&](int nt, std::vector<float>& h) {
        std::vector<float> xn((size_t) nt * D), moe((size_t) nt * D), rows((size_t) nt * K * D), sc((size_t) NE);
        std::vector<int32_t> ids((size_t) nt * K);
        std::vector<float> wts((size_t) nt * K);
        std::vector<uint8_t> mask((size_t) nt * K, 1);
        for (int l = 0; l < L; ++l) {
            for (int t = 0; t < nt; ++t) {
                rms_norm(&h[(size_t) t * D], &xn[(size_t) t * D], D);
                for (int e = 0; e < NE; ++e) {
                    float s = 0.0f;
                    const float* g = &gate[((size_t) l * NE + e) * D];
                    for (int d = 0; d < D; ++d) s += g[d] * xn[(size_t) t * D + d];
                    sc[(size_t) e] = s + (hot_bias.empty() ? 0.0f : hot_bias[(size_t) l * NE + e]);
                }
                std::vector<int> ord((size_t) NE);
                for (int e = 0; e < NE; ++e) ord[(size_t) e] = e;
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sc[(size_t) a] > sc[(size_t) b]; });
                float mx = sc[(size_t) ord[0]], sum = 0.0f;
                for (int j = 0; j < K; ++j) { wts[(size_t) t * K + j] = std::exp(sc[(size_t) ord[(size_t) j]] - mx); sum += wts[(size_t) t * K + j]; }
                for (int j = 0; j < K; ++j) { wts[(size_t) t * K + j] /= sum; ids[(size_t) t * K + j] = ord[(size_t) j]; }
                if (!counts.empty()) for (int j = 0; j < K; ++j) ++counts[(size_t) l * NE + (size_t) ord[(size_t) j]];
            }
            // the routed experts: the unit under test
            if (mode == "local") {
                if (!host.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), mask.data(), xn.data(), rows.data(), err)) { std::fprintf(stderr, "local: %s\n", err.c_str()); std::exit(1); }
            } else {
                // as the engine's Ents: one item per expert (<= 16 tokens), slots in order; then what did not come back runs here
                std::map<int, std::vector<int>> by;
                for (int i = 0; i < nt * K; ++i) by[ids[(size_t) i]].push_back(i);
                std::vector<Remote::Item> items;
                std::vector<int> slot_entry;
                for (auto& kv : by) {
                    Remote::Item it; it.e = kv.first; it.n = (int) kv.second.size(); it.slot0 = (int) slot_entry.size();
                    for (int q = 0; q < it.n; ++q) { it.tok[q] = kv.second[(size_t) q] / K; slot_entry.push_back(kv.second[(size_t) q]); }
                    items.push_back(it);
                }
                std::vector<float> y(slot_entry.size() * (size_t) D);
                std::vector<char> done;
                std::vector<uint8_t> m2((size_t) nt * K, 0);
                bool any_local = false;
                auto mark_local = [&](size_t q) {
                    for (int i = 0; i < items[q].n; ++i) m2[(size_t) slot_entry[(size_t) (items[q].slot0 + i)]] = 1;
                    any_local = true;
                };
                auto run_local = [&](bool charge) {
                    if (charge && local_delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(local_delay_ms));
                    if (any_local && !host.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), m2.data(), xn.data(), rows.data(), err)) { std::fprintf(stderr, "fallback: %s\n", err.c_str()); std::exit(1); }
                    std::fill(m2.begin(), m2.end(), (uint8_t) 0);
                    any_local = false;
                };
                if (overlap) {
                    std::vector<char> pend;
                    remote.begin(l, xn.data(), (size_t) D, items.data(), (int) items.size(), pend);
                    for (size_t q = 0; q < items.size(); ++q) if (!pend[q]) mark_local(q);
                    run_local(true);                       // the conductor's own work while the servers answer
                    remote.finish(y.data(), done);
                    for (size_t q = 0; q < items.size(); ++q) if (pend[q] && !done[q]) mark_local(q);   // a host that failed
                    run_local(false);
                } else {
                    remote.run(l, xn.data(), (size_t) D, items.data(), (int) items.size(), y.data(), done);
                    for (size_t q = 0; q < items.size(); ++q) if (!done[q]) mark_local(q);
                    run_local(true);
                }
                for (size_t q = 0; q < items.size(); ++q)
                    if (done[q])
                        for (int i = 0; i < items[q].n; ++i)
                            std::memcpy(&rows[(size_t) slot_entry[(size_t) (items[q].slot0 + i)] * D], &y[(size_t) (items[q].slot0 + i) * D], (size_t) D * 4);
            }
            rows_total += nt * K;
            row_hash = fnv(row_hash, rows.data(), rows.size() * 4);
            for (int t = 0; t < nt; ++t) {
                float* mo = &moe[(size_t) t * D];
                for (int d = 0; d < D; ++d) mo[d] = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const float w = wts[(size_t) t * K + j];
                    const float* r = &rows[((size_t) t * K + j) * D];
                    for (int d = 0; d < D; ++d) mo[d] += w * r[d];
                }
                // shared expert (dense, silu)
                std::vector<float> hid((size_t) SH);
                for (int s = 0; s < SH; ++s) {
                    float a = 0.0f;
                    const float* w1 = &sh1[((size_t) l * SH + s) * D];
                    for (int d = 0; d < D; ++d) a += w1[d] * xn[(size_t) t * D + d];
                    hid[(size_t) s] = a / (1.0f + std::exp(-a));
                }
                for (int d = 0; d < D; ++d) {
                    float a = 0.0f;
                    const float* w2 = &sh2[((size_t) l * D + d) * SH];
                    for (int s = 0; s < SH; ++s) a += w2[s] * hid[(size_t) s];
                    mo[d] += a;
                }
                std::vector<float> mn((size_t) D);
                rms_norm(mo, mn.data(), D);
                for (int d = 0; d < D; ++d) { if (!std::isfinite(mn[(size_t) d])) flag |= 1; h[(size_t) t * D + d] += 0.5f * mn[(size_t) d]; }
            }
        }
    };
    auto next_token = [&](const float* hrow) {
        std::vector<float> hn((size_t) D);
        rms_norm(hrow, hn.data(), D);
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < vocab; ++v) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += outw[(size_t) v * D + d] * hn[(size_t) d];
            if (s > bv) { bv = s; best = v; }
        }
        return best;
    };
    // the prompt as one window, then one token a step
    std::vector<float> h((size_t) n_prompt * D);
    for (int t = 0; t < n_prompt; ++t) std::memcpy(&h[(size_t) t * D], &emb[(size_t) prompt[(size_t) t] * D], (size_t) D * 4);
    forward(n_prompt, h);
    int tok = next_token(&h[(size_t) (n_prompt - 1) * D]);
    seq.push_back(tok);
    // (a recurrent carry stands in for attention's memory, so the toy does not fall into a one-token cycle)
    std::vector<float> carry(&h[(size_t) (n_prompt - 1) * D], &h[(size_t) (n_prompt - 1) * D] + D);
    const auto t_dec = std::chrono::steady_clock::now();
    for (int s = 1; s < n_new; ++s) {
        std::vector<float> h1(&emb[(size_t) tok * D], &emb[(size_t) tok * D] + D);
        for (int d = 0; d < D; ++d) h1[(size_t) d] += 0.8f * carry[(size_t) d];
        forward(1, h1);
        carry = h1;
        tok = next_token(h1.data());
        seq.push_back(tok);
    }
    const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    if (!count_out.empty()) {
        std::FILE* cf = std::fopen(count_out.c_str(), "wb");
        if (cf == nullptr || std::fwrite(counts.data(), 4, counts.size(), cf) != counts.size()) { std::fprintf(stderr, "--count-out: cannot write %s\n", count_out.c_str()); return 1; }
        std::fclose(cf);
    }
    if (show_time) std::printf("DECODE %.3f ms a token over %d tokens\n", n_new > 1 ? dec_ms / (double) (n_new - 1) : 0.0, n_new - 1);
    std::printf("TOKENS");
    for (int t : seq) std::printf(" %d", t);
    std::printf("\nROWHASH %016llx\nROWS %ld FINITE %d\n", (unsigned long long) row_hash, rows_total, (int) !(flag & 1));
    if (mode == "remote") std::printf("STATS %s\n", remote.stats().c_str());
    if (host.calls() > 0)
        std::fprintf(stderr, "HOST %llu calls, %.3f ms a call (this process's own lane)\n", (unsigned long long) host.calls(), host.compute_ms() / (double) host.calls());
    return 0;
}
