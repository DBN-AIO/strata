// src/core/net_experts_test.cpp - the pure parts of the remote-experts protocol (header-only, CPU, no sockets):
// SHA-256 / HMAC-SHA256 against FIPS 180 / RFC 4231 vectors, strict address and range parsing, and the server's
// request validation (the cases that used to poison the server: an expert id out of range, a wrong k, an oversized window).
#include "strata/core/net_proto.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::core::netproto;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static std::string sha_hex(const std::string& s) {
    uint8_t o[32];
    sha256(s.data(), s.size(), o);
    return hex(o, 32);
}
static std::string hmac_hex(const std::string& key, const std::string& msg) {
    uint8_t o[32];
    hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), nullptr, 0, o);
    return hex(o, 32);
}

// a REQ payload for n_tok x k with n_embd 4 (the validator only needs the sizes)
static std::vector<uint8_t> make_req(uint32_t layer, uint32_t n_tok, uint32_t k, uint32_t arith, const std::vector<int32_t>& ids,
                                     const std::vector<uint8_t>& mask, int64_t n_embd, size_t extra = 0) {
    std::vector<uint8_t> p(24);
    uint64_t seq = 7;
    std::memcpy(p.data(), &layer, 4); std::memcpy(p.data() + 4, &n_tok, 4); std::memcpy(p.data() + 8, &k, 4);
    std::memcpy(p.data() + 12, &arith, 4); std::memcpy(p.data() + 16, &seq, 8);
    const size_t n = (size_t) n_tok * k, mask_len = (n + 3) / 4 * 4;
    p.resize(24 + n * 4 + mask_len + (size_t) n_tok * (size_t) n_embd * 4 + extra, 0);
    std::memcpy(p.data() + 24, ids.data(), std::min(ids.size(), n) * 4);
    std::memcpy(p.data() + 24 + n * 4, mask.data(), std::min(mask.size(), n));
    return p;
}

int main() {
    // ---- SHA-256 / HMAC
    check(sha_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    check(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    check(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "sha256 two blocks");
    check(hmac_hex(std::string(20, '\x0b'), "Hi There") == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7", "hmac rfc4231 #1");
    check(hmac_hex("Jefe", "what do ya want for nothing?") == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843", "hmac rfc4231 #2");
    check(hmac_hex(std::string(131, '\xaa'), "Test Using Larger Than Block-Size Key - Hash Key First") ==
              "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54", "hmac rfc4231 #6 (key longer than a block)");
    {   // the two-part message equals the joined one
        uint8_t a[32], b[32];
        hmac_sha256("key", 3, "ab", 2, "cd", 2, a);
        hmac_sha256("key", 3, "abcd", 4, nullptr, 0, b);
        check(ct_equal(a, b, 32), "hmac two-part == joined");
        b[31] ^= 1;
        check(!ct_equal(a, b, 32), "ct_equal sees the last byte");
    }

    // ---- strict parsing
    std::string h, why;
    int port = 0, lo = 0, hi = 0;
    check(parse_hostport("127.0.0.1:7840", "x", 1, true, h, port, why) && h == "127.0.0.1" && port == 7840, "host:port");
    check(parse_hostport("7841", "127.0.0.1", 7840, true, h, port, why) && h == "127.0.0.1" && port == 7841, "bare port (listen)");
    check(parse_hostport("0.0.0.0:7840", "127.0.0.1", 7840, true, h, port, why) && h == "0.0.0.0", "explicit 0.0.0.0");
    check(parse_hostport("rig", "", 7840, false, h, port, why) && h == "rig" && port == 7840, "host only (connect): default port");
    check(parse_hostport("[::1]:7842", "", 7840, false, h, port, why) && h == "::1" && port == 7842, "[v6]:port");
    check(parse_hostport("[fe80::1]", "", 7840, false, h, port, why) && h == "fe80::1" && port == 7840, "[v6] default port");
    check(!parse_hostport("::1", "", 7840, false, h, port, why), "unbracketed v6 refused");
    check(!parse_hostport("host:0", "", 7840, false, h, port, why), "port 0 refused");
    check(!parse_hostport("host:65536", "", 7840, false, h, port, why), "port 65536 refused (no truncation)");
    check(!parse_hostport("host:78x0", "", 7840, false, h, port, why), "garbage port refused (atoi gave 78)");
    check(!parse_hostport("host:-1", "", 7840, false, h, port, why), "negative port refused");
    check(!parse_hostport("70000", "127.0.0.1", 7840, true, h, port, why), "bare out of range port refused");
    check(!parse_hostport("[::1", "", 7840, false, h, port, why), "unterminated [");
    check(!parse_hostport(":7840", "", 7840, false, h, port, why), "empty host refused");
    check(!parse_hostport("", "", 7840, false, h, port, why), "empty spec refused");
    check(parse_range("24-47", 0, 47, lo, hi) && lo == 24 && hi == 47, "range 24-47");
    check(parse_range("5", 0, 47, lo, hi) && lo == 5 && hi == 5, "single layer");
    check(!parse_range("47-24", 0, 47, lo, hi), "reversed range");
    check(!parse_range("24-48", 0, 47, lo, hi), "range past the last layer");
    check(!parse_range("a-3", 0, 47, lo, hi) && !parse_range("3-", 0, 47, lo, hi) && !parse_range("", 0, 47, lo, hi), "garbage ranges");
    check(!parse_range("-3", 0, 47, lo, hi), "leading dash");

    // ---- request validation
    ReqLimits L;
    L.layer_lo = 24; L.layer_hi = 47; L.n_embd = 4; L.n_expert = 512; L.top_k = 3; L.max_tok = 8; L.max_entries = 128;
    ReqView v;
    const std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6};          // 2 tokens x k=3
    const std::vector<uint8_t> mask = {1, 0, 1, 1, 0, 1};
    check(parse_req(L, make_req(30, 2, 3, 0, ids, mask, 4).data(), make_req(30, 2, 3, 0, ids, mask, 4).size(), v).empty() && v.rows == 4 &&
              v.n == 6 && v.ids[5] == 6 && v.mask[2] == 1, "a good request is accepted and its rows counted");
    auto bad = [&](const std::vector<uint8_t>& p, const char* what) { check(!parse_req(L, p.data(), p.size(), v).empty(), what); };
    bad(make_req(30, 2, 3, 0, {1, 2, 512, 4, 5, 6}, mask, 4), "id == n_expert refused");
    bad(make_req(30, 2, 3, 0, {1, 2, -1, 4, 5, 6}, mask, 4), "negative id refused");
    bad(make_req(30, 2, 3, 0, {1, 2, 3, 4, 5, 100000}, mask, 4), "huge id refused");
    bad(make_req(30, 2, 3, 0, ids, {1, 0, 2, 1, 0, 1}, 4), "mask byte 2 refused");
    bad(make_req(30, 2, 4, 0, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 1, 1}, 4), "k != top-k refused");
    bad(make_req(30, 9, 3, 0, std::vector<int32_t>(27, 1), std::vector<uint8_t>(27, 1), 4), "n_tok beyond the window refused");
    bad(make_req(30, 0, 3, 0, {}, {}, 4), "n_tok 0 refused");
    bad(make_req(23, 2, 3, 0, ids, mask, 4), "layer below the served range refused");
    bad(make_req(48, 2, 3, 0, ids, mask, 4), "layer above the served range refused");
    bad(make_req(30, 2, 3, 5, ids, mask, 4), "unknown arithmetic mode refused");
    bad(make_req(30, 2, 3, 0, ids, mask, 4, 4), "payload longer than its header says refused");
    {
        std::vector<uint8_t> p = make_req(30, 2, 3, 0, ids, mask, 4);
        p.pop_back();
        bad(p, "truncated payload refused");
        p.resize(10);
        bad(p, "payload shorter than the header refused");
    }
    {   // k = 3 but a window of 50 tokens would overflow the tables: n_tok bound first, entries bound second
        ReqLimits L2 = L; L2.max_tok = 64; L2.max_entries = 128;
        std::vector<uint8_t> p = make_req(30, 50, 3, 0, std::vector<int32_t>(150, 1), std::vector<uint8_t>(150, 1), 4);
        check(!parse_req(L2, p.data(), p.size(), v).empty(), "n_tok*k beyond the tables refused");
    }

    {   // ---- B2: the pairs a GPU expert server's helpers hold (served layers only, profile order first, then the rest)
        const std::vector<std::pair<int32_t, int32_t>> ranked = {{5, 1}, {2, 3}, {3, 0}, {2, 3}, {4, 9}, {3, 2}, {9, 0}, {2, 1}};
        const auto sp = served_pairs(ranked, 6, 4, 2, 3);   // 6 layers x 4 experts, served layers 2-3
        check(sp.size() == 8, "served_pairs: every served pair exactly once (2 layers x 4 experts)");
        check(sp.size() >= 4 && sp[0] == std::make_pair(2, 3) && sp[1] == std::make_pair(3, 0) && sp[2] == std::make_pair(3, 2) &&
              sp[3] == std::make_pair(2, 1), "served_pairs: profile order first, repeats and other layers dropped");
        bool in_range = true;
        for (const auto& p : sp) in_range = in_range && p.first >= 2 && p.first <= 3 && p.second >= 0 && p.second < 4;
        check(in_range, "served_pairs: nothing outside the served layers or the expert count");
        check(sp.size() == 8 && sp[4] == std::make_pair(2, 0) && sp[5] == std::make_pair(3, 1) && sp[6] == std::make_pair(2, 2) &&
              sp[7] == std::make_pair(3, 3), "served_pairs: the unranked rest expert-major");
        check(served_pairs(ranked, 6, 4, 3, 2).empty() && served_pairs(ranked, 6, 4, 0, 6).empty(), "served_pairs: a bad range gives nothing");
    }

    {   // ---- B3: the windows a prompt goes to the server in
        std::vector<uint8_t> m(14 * 10, 0);   // 14 tokens x 10 entries; window limit 8
        m[3] = 1; m[79] = 1; m[80] = 1; m[139] = 1;
        auto w = prompt_windows(14, 10, m.data(), 8);
        check(w.size() == 2 && w[0].t0 == 0 && w[0].nt == 8 && w[0].want == 2 && w[1].t0 == 8 && w[1].nt == 6 && w[1].want == 2,
              "prompt_windows: 14 tokens in windows of 8 and 6, entries counted per window");
        std::fill(m.begin(), m.begin() + 80, 0);
        w = prompt_windows(14, 10, m.data(), 8);
        check(w.size() == 1 && w[0].t0 == 8 && w[0].want == 2, "prompt_windows: a window with nothing asked for is left out");
        check(prompt_windows(0, 10, m.data(), 8).empty() && prompt_windows(14, 10, m.data(), 0).empty(), "prompt_windows: empty input");
        std::vector<uint8_t> all(16 * 10, 1);
        w = prompt_windows(16, 10, all.data(), 8);
        check(w.size() == 2 && w[1].t0 == 8 && w[1].nt == 8 && w[0].want == 80, "prompt_windows: exact multiple of the window");
    }

    {   // ---- B5: routes with expert ranges, the owner rule, coverage, the server's expert range
        RouteSpec a, b, c;
        std::string w;
        check(parse_route_item("127.0.0.1:7841=38-47/0-255", 48, 512, a, w) && a.host == "127.0.0.1" && a.port == 7841 && a.lo == 38 &&
              a.hi == 47 && a.elo == 0 && a.ehi == 255, "route item with an expert range");
        check(parse_route_item("rig=38-47", 48, 512, b, w) && b.port == 7840 && b.elo == 0 && b.ehi == 511, "route item without: every expert");
        check(!parse_route_item("rig=38-47/0-512", 48, 512, c, w), "expert range past n_expert refused");
        check(!parse_route_item("rig=38-47/x", 48, 512, c, w) && !parse_route_item("rig=38-48", 48, 512, c, w), "bad ranges refused");
        check(routes_conflict(a, b), "same layers, overlapping experts conflict");
        check(parse_route_item("[::1]:7850=38-47/256-511", 48, 512, c, w) && c.host == "::1" && !routes_conflict(a, c),
              "same layers, disjoint experts do not conflict");
        check(layer_covered({a, c}, 40, 512) && !layer_covered({a}, 40, 512) && !layer_covered({a, c}, 37, 512), "coverage per layer");
        ReqLimits L3; L3.layer_lo = 24; L3.layer_hi = 47; L3.n_embd = 4; L3.n_expert = 512; L3.top_k = 3; L3.max_tok = 8;
        L3.max_entries = 128; L3.expert_lo = 256; L3.expert_hi = 511;
        ReqView v3;
        auto p_in = make_req(30, 2, 3, 0, {256, 300, 511, 256, 1, 2}, {1, 1, 1, 1, 0, 0}, 4);
        check(parse_req(L3, p_in.data(), p_in.size(), v3).empty(), "server expert range: owned experts asked for, others unmasked");
        auto p_out = make_req(30, 2, 3, 0, {256, 300, 511, 256, 1, 2}, {1, 1, 1, 1, 1, 0}, 4);
        check(!parse_req(L3, p_out.data(), p_out.size(), v3).empty(), "server expert range: a masked expert outside it refused");
        const auto sp = served_pairs({{38, 300}, {38, 5}, {39, 256}}, 48, 512, 38, 39, 256, 511);
        check(sp.size() == 512 && sp[0] == std::make_pair(38, 300) && sp[1] == std::make_pair(39, 256), "served_pairs with an expert range");
    }

    if (g_fail == 0) std::printf("net_experts_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
