// include/strata/core/net_proto.hpp - the pure (no socket, no CUDA) parts of the remote-experts protocol:
// SHA-256 / HMAC-SHA256 for the shared-secret handshake, strict host:port / layer-range parsing, and the server's
// request validation.  Header-only so the unit test (src/core/net_experts_test.cpp) builds without the engine.
//
// Everything on the wire is host byte order = little endian, fp32/fp64 IEEE: asserted here, not assumed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::netproto {

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "SXP wire format is little endian; add byte swaps before porting to big endian");
#endif
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559, "SXP carries IEEE-754 fp32/fp64");
static_assert(sizeof(float) == 4 && sizeof(double) == 8, "SXP float sizes");

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4) and HMAC (RFC 2104)
struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint8_t buf[64] = {};
    uint64_t total = 0;
    size_t fill = 0;

    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void* data, size_t len) {
        const uint8_t* p = (const uint8_t*) data;
        total += len;
        while (len > 0) {
            const size_t take = len < 64 - fill ? len : 64 - fill;
            std::memcpy(buf + fill, p, take);
            fill += take; p += take; len -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    void finish(uint8_t out[32]) {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(len, 8);
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = (uint8_t) (h[i] >> 24); out[4 * i + 1] = (uint8_t) (h[i] >> 16);
            out[4 * i + 2] = (uint8_t) (h[i] >> 8); out[4 * i + 3] = (uint8_t) h[i];
        }
    }
};

inline void sha256(const void* data, size_t len, uint8_t out[32]) {
    Sha256 c;
    c.update(data, len);
    c.finish(out);
}

/// HMAC-SHA256(key, msg1 || msg2): the two-part message saves a copy for the handshake (nonce || hello fields).
inline void hmac_sha256(const void* key, size_t klen, const void* m1, size_t l1, const void* m2, size_t l2, uint8_t out[32]) {
    uint8_t k[64] = {};
    if (klen > 64) sha256(key, klen, k);
    else if (klen > 0) std::memcpy(k, key, klen);
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) { ipad[i] = (uint8_t) (k[i] ^ 0x36); opad[i] = (uint8_t) (k[i] ^ 0x5c); }
    uint8_t inner[32];
    Sha256 a;
    a.update(ipad, 64);
    if (l1 > 0) a.update(m1, l1);
    if (l2 > 0) a.update(m2, l2);
    a.finish(inner);
    Sha256 b;
    b.update(opad, 64);
    b.update(inner, 32);
    b.finish(out);
}

/// Constant-time equality (no early exit on the first differing byte).
inline bool ct_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc = (uint8_t) (acc | (a[i] ^ b[i]));
    return acc == 0;
}

inline std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}

// ---------------------------------------------------------------- strict number / address parsing
/// A decimal integer in [lo, hi], digits only (no sign, no space, no trailing text): false otherwise.
inline bool parse_uint(const std::string& s, long lo, long hi, long& out) {
    if (s.empty() || s.size() > 9) return false;
    long v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    if (v < lo || v > hi) return false;
    out = v;
    return true;
}

/// "a-b" or "a": an inclusive layer range, both ends digits.
inline bool parse_range(const std::string& s, long lo_limit, long hi_limit, int& lo, int& hi) {
    const size_t dash = s.find('-');
    long a = 0, b = 0;
    if (!parse_uint(dash == std::string::npos ? s : s.substr(0, dash), lo_limit, hi_limit, a)) return false;
    if (dash == std::string::npos) b = a;
    else if (!parse_uint(s.substr(dash + 1), lo_limit, hi_limit, b)) return false;
    if (b < a) return false;
    lo = (int) a; hi = (int) b;
    return true;
}

/// "host:port", "[v6addr]:port", "host" / "[v6addr]" (default port), and - with bare_is_port - "port" alone (default
/// host).  An IPv6 literal must be bracketed (an unbracketed one has several colons and is refused).  The port is 1..65535,
/// digits only.  On failure `err` says why.
inline bool parse_hostport(const std::string& spec, const std::string& def_host, int def_port, bool bare_is_port,
                           std::string& host, int& port, std::string& err) {
    long p = def_port;
    std::string h;
    if (spec.empty()) { err = "empty address"; return false; }
    if (spec[0] == '[') {
        const size_t close = spec.find(']');
        if (close == std::string::npos || close == 1) { err = "bad [ipv6] literal in '" + spec + "'"; return false; }
        h = spec.substr(1, close - 1);
        const std::string rest = spec.substr(close + 1);
        if (!rest.empty()) {
            if (rest[0] != ':' || !parse_uint(rest.substr(1), 1, 65535, p)) { err = "bad port in '" + spec + "' (1..65535)"; return false; }
        }
    } else {
        const size_t colon = spec.find(':');
        if (colon == std::string::npos) {
            long v = 0;
            if (bare_is_port && parse_uint(spec, 1, 65535, v)) { h = def_host; p = v; }
            else if (bare_is_port && spec.find_first_not_of("0123456789") == std::string::npos) {
                err = "bad port in '" + spec + "' (1..65535)"; return false;
            } else h = spec;
        } else {
            if (spec.find(':', colon + 1) != std::string::npos) { err = "an IPv6 literal needs [brackets]: '" + spec + "'"; return false; }
            h = spec.substr(0, colon);
            if (!parse_uint(spec.substr(colon + 1), 1, 65535, p)) { err = "bad port in '" + spec + "' (1..65535)"; return false; }
        }
    }
    if (h.empty()) { err = "empty host in '" + spec + "'"; return false; }
    host = h; port = (int) p;
    return true;
}

// ---------------------------------------------------------------- the REQ frame, validated before anything runs
// REQ payload: u32 layer | u32 n_tok | u32 k | u32 arith | u64 seq | i32 ids[n_tok*k] | u8 mask[n_tok*k, padded to 4] |
//              f32 x[n_tok*n_embd]
struct ReqLimits {
    int layer_lo = 0, layer_hi = -1;       ///< served layers (inclusive)
    int expert_lo = 0, expert_hi = -1;     ///< served experts (inclusive, --serve-experts); hi < 0: every expert
    int64_t n_embd = 0, n_expert = 0, top_k = 0;
    int64_t max_tok = 0, max_entries = 0;  ///< the engine's window: rows, and rows x k
};
struct ReqView {
    uint32_t layer = 0, n_tok = 0, k = 0, arith = 0;
    uint64_t seq = 0;
    size_t n = 0, mask_len = 0;
    const int32_t* ids = nullptr;
    const uint8_t* mask = nullptr;
    const float* x = nullptr;
    uint32_t rows = 0;                     ///< entries with mask 1
};
inline constexpr size_t kReqHead = 24;

/// Empty string: `v` is filled and every field is inside the limits.  Otherwise the reason (nothing was run).
inline std::string parse_req(const ReqLimits& L, const uint8_t* p, size_t len, ReqView& v) {
    if (len < kReqHead) return "request shorter than its header";
    std::memcpy(&v.layer, p, 4); std::memcpy(&v.n_tok, p + 4, 4); std::memcpy(&v.k, p + 8, 4);
    std::memcpy(&v.arith, p + 12, 4); std::memcpy(&v.seq, p + 16, 8);
    if ((int64_t) v.layer < L.layer_lo || (int64_t) v.layer > L.layer_hi)
        return "layer " + std::to_string(v.layer) + " is outside the served layers " + std::to_string(L.layer_lo) + "-" + std::to_string(L.layer_hi);
    if (v.n_tok < 1 || (int64_t) v.n_tok > L.max_tok) return "n_tok " + std::to_string(v.n_tok) + " outside 1.." + std::to_string(L.max_tok);
    if ((int64_t) v.k != L.top_k) return "k " + std::to_string(v.k) + " differs from the model's top-k " + std::to_string(L.top_k);
    if ((int64_t) v.n_tok * (int64_t) v.k > L.max_entries) return "n_tok*k exceeds the window tables";
    if (v.arith > 1) return "unknown arithmetic mode " + std::to_string(v.arith);
    v.n = (size_t) v.n_tok * v.k;
    v.mask_len = (v.n + 3) / 4 * 4;
    if (len != kReqHead + v.n * 4 + v.mask_len + (size_t) v.n_tok * (size_t) L.n_embd * 4) return "payload size does not match n_tok / k";
    v.ids = (const int32_t*) (p + kReqHead);
    v.mask = p + kReqHead + v.n * 4;
    v.x = (const float*) (p + kReqHead + v.n * 4 + v.mask_len);
    v.rows = 0;
    for (size_t i = 0; i < v.n; ++i) {
        int32_t id;
        std::memcpy(&id, p + kReqHead + i * 4, 4);
        if (id < 0 || id >= L.n_expert) return "expert id " + std::to_string(id) + " outside 0.." + std::to_string(L.n_expert - 1) + " (entry " + std::to_string(i) + ")";
        if (v.mask[i] > 1) return "mask byte " + std::to_string(v.mask[i]) + " is not 0 or 1 (entry " + std::to_string(i) + ")";
        if (v.mask[i] && L.expert_hi >= 0 && (id < L.expert_lo || id > L.expert_hi))
            return "expert " + std::to_string(id) + " is not served here (experts " + std::to_string(L.expert_lo) + "-" +
                   std::to_string(L.expert_hi) + ", entry " + std::to_string(i) + ")";
        v.rows += v.mask[i];
    }
    return std::string();
}

// ---------------------------------------------------------------- the prompt path (B3)
/// One REQ of a prompt: tokens [t0, t0 + nt) and the number of asked-for entries in them.
struct PromptWindow { int64_t t0 = 0, nt = 0; uint32_t want = 0; };
/// The windows a prompt of T tokens (k entries each, mask[i] != 0 = asked for) is sent in: at most `max_tok` tokens each (the
/// server's window limit), in order, a window with nothing asked for left out.
inline std::vector<PromptWindow> prompt_windows(int64_t T, int64_t k, const uint8_t* mask, int64_t max_tok) {
    std::vector<PromptWindow> out;
    if (T <= 0 || k <= 0 || max_tok <= 0) return out;
    for (int64_t t0 = 0; t0 < T; t0 += max_tok) {
        PromptWindow w;
        w.t0 = t0;
        w.nt = T - t0 < max_tok ? T - t0 : max_tok;
        for (int64_t i = t0 * k; i < (t0 + w.nt) * k; ++i) w.want += mask[i] ? 1u : 0u;
        if (w.want > 0) out.push_back(w);
    }
    return out;
}

// ---------------------------------------------------------------- GPU expert server (--expert-server-gpu)
/// The (layer, expert) pairs an expert server's helper GPUs are filled with, best first: the ranked (profile) pairs of the
/// served layers lo..hi in their order, each once, then every served pair the ranking does not name, expert-major (so a
/// fill that runs out of room spreads its misses across the layers).  Pairs outside lo..hi or 0..n_expert-1 are dropped:
/// a server computes only its own layers, and a helper slot spent on another layer is a served expert left on the CPU.
/// B5: `elo`..`ehi` (default every expert) limits it to the experts this server owns (--serve-experts).
inline std::vector<std::pair<int32_t, int32_t>> served_pairs(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                                             int64_t n_layers, int64_t n_expert, int lo, int hi,
                                                             int elo = 0, int ehi = -1) {
    std::vector<std::pair<int32_t, int32_t>> out;
    if (ehi < 0) ehi = (int) n_expert - 1;
    if (n_layers <= 0 || n_expert <= 0 || lo < 0 || hi < lo || hi >= n_layers || elo < 0 || ehi < elo || ehi >= n_expert) return out;
    std::vector<uint8_t> seen((size_t) n_layers * (size_t) n_expert, 0);
    for (const auto& pr : ranked) {
        if (pr.first < lo || pr.first > hi || pr.second < elo || pr.second > ehi) continue;
        uint8_t& s = seen[(size_t) pr.first * (size_t) n_expert + (size_t) pr.second];
        if (s) continue;
        s = 1;
        out.push_back(pr);
    }
    for (int64_t e = elo; e <= ehi; ++e)
        for (int64_t l = lo; l <= hi; ++l)
            if (!seen[(size_t) l * (size_t) n_expert + (size_t) e]) out.emplace_back((int32_t) l, (int32_t) e);
    return out;
}

// ---------------------------------------------------------------- B5: several hosts per layer, by expert range
/// One --remote-experts item: host:port=lo-hi[/e0-e1] (layers lo..hi; experts e0..e1, default every expert).
struct RouteSpec { std::string host; int port = 7840; int lo = 0, hi = 0, elo = 0, ehi = 0; };
inline bool parse_route_item(const std::string& item, int64_t n_layers, int64_t n_expert, RouteSpec& r, std::string& why) {
    const size_t eq = item.rfind('=');
    if (eq == std::string::npos) { why = "item needs host:port=lo-hi[/e0-e1]: " + item; return false; }
    std::string lh = item.substr(eq + 1), er;
    const size_t sl = lh.find('/');
    if (sl != std::string::npos) { er = lh.substr(sl + 1); lh = lh.substr(0, sl); }
    if (!parse_hostport(item.substr(0, eq), "", 7840, false, r.host, r.port, why)) return false;
    if (!parse_range(lh, 0, (long) n_layers - 1, r.lo, r.hi)) { why = "layer range out of range: " + item; return false; }
    r.elo = 0;
    r.ehi = (int) n_expert - 1;
    if (sl != std::string::npos && !parse_range(er, 0, (long) n_expert - 1, r.elo, r.ehi)) { why = "expert range out of range: " + item; return false; }
    return true;
}
/// Two routes conflict when they share a layer AND an expert (each (layer, expert) has one owner).
inline bool routes_conflict(const RouteSpec& a, const RouteSpec& b) {
    return !(a.hi < b.lo || a.lo > b.hi) && !(a.ehi < b.elo || a.elo > b.ehi);
}
/// The routes of `layer` together own every one of its n_expert experts.
inline bool layer_covered(const std::vector<RouteSpec>& rs, int64_t layer, int64_t n_expert) {
    if (n_expert <= 0) return false;
    std::vector<uint8_t> own((size_t) n_expert, 0);
    for (const RouteSpec& r : rs)
        if (layer >= r.lo && layer <= r.hi)
            for (int e = r.elo < 0 ? 0 : r.elo; e <= r.ehi && e < n_expert; ++e) own[(size_t) e] = 1;
    for (uint8_t o : own) if (!o) return false;
    return true;
}

}  // namespace strata::core::netproto