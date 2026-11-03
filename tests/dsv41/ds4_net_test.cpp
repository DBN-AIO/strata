// Strata. The synthetic split test's program for DeepSeek experts over the LAN (tools/dsv41/synthetic_split_test.py).
//
// A tiny DeepSeek-V4.1-shaped model: layers of routed experts (n_exp, top-k, gate / up / down in the quant types the real
// sets use, from a pack that tools/dsv41/make_synthetic_pack.py wrote), a router, a shared expert and a residual, driven
// greedily for N tokens from a prompt that goes through as one window.  The experts are computed
//   --mode local   in this process (net::ExpertHost: the code the server runs)
//   --mode remote  by expert servers (net::Remote, the conductor's side), with this process's own host as the fallback
// and the program prints the generated tokens and a hash of every expert row it received (bit identity, not just argmax).
// Equal lines from the two modes, with a server on localhost or on another PC, is the test.
//
// --causal: the toy's memory is the previous TOKEN (not a carried hidden state), so a window of tokens is exactly what a sequence of
//   single steps would give, as with attention; the prompt goes through in windows of kMaxTok, any length (--prompt N, --prompt-seed S
//   makes a unique prefix).  --spec K adds speculative decoding on it (greedy, so the tokens must equal plain decoding's): each round a
//   drafter proposes up to K tokens, ONE window of 1 + K tokens goes through every layer (through the servers in --mode remote), and the
//   longest matching prefix plus one more token is kept.  --draft-acc P: an oracle drafter right with probability P per draft (the
//   accept rate is then a knob); without it a prompt-lookup n-gram drafter.  The weights are random: the accept rate says nothing about a
//   real drafter, the call counts and times are the mechanism's.
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
    bool expect_open_fail = false, causal = false;
    int spec_k = 0;
    double draft_acc = -1.0;
    unsigned prompt_seed = 1;
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
        else if (a == "--causal") causal = true;
        else if (a == "--spec") { spec_k = std::atoi(v().c_str()); causal = true; }
        else if (a == "--draft-acc") draft_acc = std::atof(v().c_str());
        else if (a == "--prompt-seed") prompt_seed = (unsigned) std::atoi(v().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    ExpertMap map;
    std::string err;
    if (pack.empty() || !map.load(pack, err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 2; }
    const int D = map.dim, L = map.n_layers, K = map.top_k, NE = map.n_exp;
    if (n_prompt < 1 || (!causal && n_prompt > kMaxTok) || spec_k < 0 || spec_k >= kMaxTok) { std::fprintf(stderr, "--prompt 1..%d\n", kMaxTok); return 2; }

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

    uint64_t row_hash = 14695981039346656037ull, flag = 0;
    long rows_total = 0;
    uint64_t windows = 0;
    bool counting = true;                      // false for the oracle's look-ahead pass (not hashed, not counted)
    bool use_remote = mode == "remote";
    std::vector<int> seq;
    std::vector<int> prompt;
    for (int i = 0; i < n_prompt; ++i) prompt.push_back((int) ((i * 7 + 3) % vocab));

    // one window of nt tokens (their hidden states h: nt x D) through every layer
    auto forward = [&](int nt, std::vector<float>& h) {
        if (counting) ++windows;
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
                    sc[(size_t) e] = s;
                }
                std::vector<int> ord((size_t) NE);
                for (int e = 0; e < NE; ++e) ord[(size_t) e] = e;
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sc[(size_t) a] > sc[(size_t) b]; });
                float mx = sc[(size_t) ord[0]], sum = 0.0f;
                for (int j = 0; j < K; ++j) { wts[(size_t) t * K + j] = std::exp(sc[(size_t) ord[(size_t) j]] - mx); sum += wts[(size_t) t * K + j]; }
                for (int j = 0; j < K; ++j) { wts[(size_t) t * K + j] /= sum; ids[(size_t) t * K + j] = ord[(size_t) j]; }
            }
            // the routed experts: the unit under test
            if (!use_remote) {
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
                remote.run(l, xn.data(), (size_t) D, items.data(), (int) items.size(), y.data(), done);
                std::vector<uint8_t> m2((size_t) nt * K, 0);
                bool any_local = false;
                for (size_t q = 0; q < items.size(); ++q) {
                    if (done[q]) {
                        for (int i = 0; i < items[q].n; ++i)
                            std::memcpy(&rows[(size_t) slot_entry[(size_t) (items[q].slot0 + i)] * D], &y[(size_t) (items[q].slot0 + i) * D], (size_t) D * 4);
                    } else {
                        for (int i = 0; i < items[q].n; ++i) m2[(size_t) slot_entry[(size_t) (items[q].slot0 + i)]] = 1;
                        any_local = true;
                    }
                }
                if (any_local && !host.compute((uint32_t) l, (uint32_t) nt, (uint32_t) K, ids.data(), m2.data(), xn.data(), rows.data(), err)) { std::fprintf(stderr, "fallback: %s\n", err.c_str()); std::exit(1); }
            }
            if (counting) {
                rows_total += nt * K;
                row_hash = fnv(row_hash, rows.data(), rows.size() * 4);
            }
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
    if (!causal) {
        // the prompt as one window, then one token a step
        std::vector<float> h((size_t) n_prompt * D);
        for (int t = 0; t < n_prompt; ++t) std::memcpy(&h[(size_t) t * D], &emb[(size_t) prompt[(size_t) t] * D], (size_t) D * 4);
        forward(n_prompt, h);
        int tok = next_token(&h[(size_t) (n_prompt - 1) * D]);
        seq.push_back(tok);
        // (a recurrent carry stands in for attention's memory, so the toy does not fall into a one-token cycle)
        std::vector<float> carry(&h[(size_t) (n_prompt - 1) * D], &h[(size_t) (n_prompt - 1) * D] + D);
        for (int s = 1; s < n_new; ++s) {
            std::vector<float> h1(&emb[(size_t) tok * D], &emb[(size_t) tok * D] + D);
            for (int d = 0; d < D; ++d) h1[(size_t) d] += 0.8f * carry[(size_t) d];
            forward(1, h1);
            carry = h1;
            tok = next_token(h1.data());
            seq.push_back(tok);
        }
    } else {
        std::mt19937 prng(prompt_seed * 2654435761u + 17u);
        std::vector<int> pr((size_t) n_prompt);
        for (int& t : pr) t = (int) (prng() % (unsigned) vocab);
        auto cinput = [&](int tk, int pv, float* o) {
            for (int d = 0; d < D; ++d) o[d] = emb[(size_t) tk * D + d] + 0.8f * emb[(size_t) pv * D + d];
        };
        // a window of tokens w (the one before w[0] is pv0): the greedy next token after each
        auto preds_of = [&](const std::vector<int>& w, int pv0, std::vector<int>& preds) {
            std::vector<float> hw(w.size() * (size_t) D);
            for (size_t j = 0; j < w.size(); ++j) cinput(w[j], j ? w[j - 1] : pv0, &hw[j * (size_t) D]);
            forward((int) w.size(), hw);
            preds.resize(w.size());
            for (size_t j = 0; j < w.size(); ++j) preds[j] = next_token(&hw[j * (size_t) D]);
        };
        std::vector<int> pw, preds;
        for (int c = 0; c < n_prompt; c += kMaxTok) {   // the prompt, a window of kMaxTok at a time
            pw.assign(pr.begin() + c, pr.begin() + std::min(n_prompt, c + kMaxTok));
            preds_of(pw, c ? pr[(size_t) c - 1] : vocab - 1, preds);
        }
        seq.push_back(preds.back());
        const int prompt_last = pr.back();
        // the oracle drafter's truth: the plain greedy continuation, computed apart from what is measured and counted
        std::vector<int> truth;
        std::mt19937 drng(12345u);
        if (spec_k > 0 && draft_acc >= 0.0) {
            counting = false;
            use_remote = false;
            truth.push_back(seq[0]);
            for (int i2 = 0; i2 < n_new + spec_k + 2; ++i2) {
                preds_of({truth.back()}, truth.size() >= 2 ? truth[truth.size() - 2] : prompt_last, preds);
                truth.push_back(preds[0]);
            }
            counting = true;
            use_remote = mode == "remote";
        }
        const uint64_t w0 = windows, c0 = mode == "remote" ? remote.calls() : 0;
        const auto t0 = std::chrono::steady_clock::now();
        long rounds = 0, drafted = 0, accepted = 0;
        while ((int) seq.size() < n_new) {
            std::vector<int> d;
            const int have = (int) seq.size();
            if (spec_k > 0 && draft_acc >= 0.0) {
                for (int j = 0; j < spec_k && have + j < (int) truth.size(); ++j) {
                    const int t = truth[(size_t) have + j];
                    const bool right = (double) (drng() % 1000000u) / 1e6 < draft_acc;
                    d.push_back(right ? t : (t + 1 + (int) (drng() % (unsigned) (vocab - 1))) % vocab);
                }
            } else if (spec_k > 0) {   // prompt lookup: the token after the latest earlier occurrence of the last two
                std::vector<int> hist(pr);
                hist.insert(hist.end(), seq.begin(), seq.end());
                const int n = (int) hist.size();
                for (int j = n - 3; j >= 0; --j)
                    if (hist[(size_t) j] == hist[(size_t) n - 2] && hist[(size_t) j + 1] == hist[(size_t) n - 1]) {
                        for (int q = j + 2; q < n && (int) d.size() < spec_k; ++q) d.push_back(hist[(size_t) q]);
                        break;
                    }
            }
            std::vector<int> w;
            w.push_back(seq.back());
            w.insert(w.end(), d.begin(), d.end());
            preds_of(w, seq.size() >= 2 ? seq[seq.size() - 2] : prompt_last, preds);
            int a = 0;
            while (a < (int) d.size() && preds[(size_t) a] == d[(size_t) a]) ++a;
            for (int j = 0; j <= a && (int) seq.size() < n_new; ++j) seq.push_back(preds[(size_t) j]);
            ++rounds;
            drafted += (long) d.size();
            accepted += a;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const uint64_t dw = windows - w0, dc = (mode == "remote" ? remote.calls() : 0) - c0;
        std::printf("SPEC k %d rounds %ld drafted %ld accepted %ld windows %llu layer_calls %llu wire_calls %llu\n", spec_k, rounds, drafted, accepted,
                    (unsigned long long) windows, (unsigned long long) windows * (unsigned) L,
                    (unsigned long long) (mode == "remote" ? remote.calls() : 0));
        std::printf("DECODE tokens %d ms %.1f tok_s %.2f windows %llu layer_calls %llu wire_calls %llu accepted_per_round %.3f\n", n_new - 1, ms,
                    (n_new - 1) * 1000.0 / ms, (unsigned long long) dw, (unsigned long long) dw * (unsigned) L, (unsigned long long) dc,
                    rounds ? (double) accepted / (double) rounds : 0.0);
    }
    std::printf("TOKENS");
    for (int t : seq) std::printf(" %d", t);
    std::printf("\nROWHASH %016llx\nROWS %ld FINITE %d\n", (unsigned long long) row_hash, rows_total, (int) !(flag & 1));
    if (mode == "remote") std::printf("STATS %s\n", remote.stats().c_str());
    if (host.calls() > 0)
        std::fprintf(stderr, "HOST %llu calls, %.3f ms a call (this process's own lane)\n", (unsigned long long) host.calls(), host.compute_ms() / (double) host.calls());
    return 0;
}
