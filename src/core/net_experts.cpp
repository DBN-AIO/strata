// src/core/net_experts.cpp - remote experts over the LAN: client (conductor side) and expert server.
// Protocol: include/strata/core/net_experts.hpp.  Sockets: the small shim below is the only OS specific code.
#include "strata/core/net_experts.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace strata::core {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int64_t H = strata::kernels::cpu::H;
constexpr uint32_t kMagic = 0x31505853u;   // "SXP1"
enum : uint32_t { kHello = 1, kHelloAck = 2, kReq = 3, kResp = 4, kErr = 5, kBye = 6 };
constexpr uint64_t kMaxFrame = 64ull << 20;

double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- socket shim (POSIX / Winsock)
#ifdef _WIN32
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
void sock_init() { static bool done = false; if (!done) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); done = true; } }
void sock_close(sock_t s) { if (s != kBad) closesocket(s); }
int sock_errno() { return WSAGetLastError(); }
#else
using sock_t = int;
constexpr sock_t kBad = -1;
void sock_init() {}
void sock_close(sock_t s) { if (s != kBad) ::close(s); }
int sock_errno() { return errno; }
#endif

void sock_tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 4 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
}

bool send_all(sock_t s, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int r = (int) ::send(s, c, (int) (n > (1u << 30) ? (1u << 30) : n), 0);
        if (r <= 0) return false;
        c += r; n -= (size_t) r;
    }
    return true;
}
bool recv_all(sock_t s, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const int r = (int) ::recv(s, c, (int) (n > (1u << 30) ? (1u << 30) : n), 0);
        if (r <= 0) return false;
        c += r; n -= (size_t) r;
    }
    return true;
}

struct Hdr { uint32_t magic, type; uint64_t len; };
bool send_frame(sock_t s, uint32_t type, const void* payload, uint64_t len) {
    Hdr h{kMagic, type, len};
    if (!send_all(s, &h, sizeof h)) return false;
    return len == 0 || send_all(s, payload, (size_t) len);
}
bool recv_frame(sock_t s, uint32_t& type, std::vector<uint8_t>& payload) {
    Hdr h;
    if (!recv_all(s, &h, sizeof h) || h.magic != kMagic || h.len > kMaxFrame) return false;
    type = h.type;
    payload.resize((size_t) h.len);
    return h.len == 0 || recv_all(s, payload.data(), (size_t) h.len);
}

sock_t sock_connect(const std::string& host, int port, std::string& err) {
    sock_init();
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return kBad;
    }
    sock_t s = kBad;
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == kBad) continue;
        if (::connect(s, a->ai_addr, (int) a->ai_addrlen) == 0) break;
        sock_close(s);
        s = kBad;
    }
    freeaddrinfo(res);
    if (s == kBad) err = "cannot connect to " + host + ":" + ps + " (errno " + std::to_string(sock_errno()) + ")";
    else sock_tune(s);
    return s;
}

template <class T> void put(std::vector<uint8_t>& b, const T& v) {
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
}
}  // namespace

// ================================================================== client
// B5 (request pipelining): a layer may be served by several hosts at once, each owning a range of experts
// (host:port=lo-hi/e0-e1).  begin() sends the layer's request to every host that owns one of its CPU entries before
// the conductor's own CPU work, finish() collects all answers: the hosts compute their shares of the same layer at the
// same time, so a layer costs the slowest host's round trip instead of one host's whole share.  The prompt path keeps
// up to STRATA_NET_PIPELINE windows (default 4) in flight per host instead of one round trip per window.
bool NetExperts::open(const std::string& spec, int64_t n_layers, int64_t n_expert, std::string& err) {
    n_expert_ = n_expert;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string item = spec.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) { err = "--remote-experts item needs host:port=lo-hi[/e0-e1]: " + item; return false; }
        Route r;
        const std::string hp = item.substr(0, eq);
        std::string lh = item.substr(eq + 1);
        const size_t slash = lh.find('/');
        r.elo = 0;
        r.ehi = (int) n_expert - 1;
        if (slash != std::string::npos) {   // the expert range this host owns for those layers
            const std::string er = lh.substr(slash + 1);
            lh = lh.substr(0, slash);
            const size_t ed = er.find('-');
            r.elo = std::atoi(er.c_str());
            r.ehi = ed == std::string::npos ? r.elo : std::atoi(er.c_str() + ed + 1);
            if (r.elo < 0 || r.ehi < r.elo || r.ehi >= n_expert) { err = "expert range out of range: " + item; return false; }
        }
        const size_t colon = hp.rfind(':');
        r.host = colon == std::string::npos ? hp : hp.substr(0, colon);
        if (colon != std::string::npos) r.port = std::atoi(hp.c_str() + colon + 1);
        const size_t dash = lh.find('-');
        r.lo = std::atoi(lh.c_str());
        r.hi = dash == std::string::npos ? r.lo : std::atoi(lh.c_str() + dash + 1);
        if (r.lo < 0 || r.hi < r.lo || r.hi >= n_layers) { err = "layer range out of range: " + item; return false; }
        for (const Route& o : routes_)   // two hosts may share layers only with disjoint expert ranges
            if (!(r.hi < o.lo || r.lo > o.hi) && !(r.ehi < o.elo || r.elo > o.ehi)) {
                err = "overlapping layer and expert ranges in --remote-experts"; return false;
            }
        const sock_t s = sock_connect(r.host, r.port, err);
        if (s == kBad) return false;
        std::vector<uint8_t> hello;
        put<uint32_t>(hello, 1); put<uint32_t>(hello, (uint32_t) H);
        put<uint32_t>(hello, (uint32_t) n_expert); put<uint32_t>(hello, (uint32_t) n_layers);
        uint32_t type = 0;
        std::vector<uint8_t> ack;
        if (!send_frame(s, kHello, hello.data(), hello.size()) || !recv_frame(s, type, ack)) {
            err = "no handshake from " + r.host; sock_close(s); return false;
        }
        if (type == kErr) { err = "expert server " + r.host + ": " + std::string(ack.begin(), ack.end()); sock_close(s); return false; }
        uint32_t a[7] = {0, 0, 0, 0, 0, 0, (uint32_t) n_expert - 1};
        if (type != kHelloAck || ack.size() < 20) { err = "bad handshake from " + r.host; sock_close(s); return false; }
        std::memcpy(a, ack.data(), std::min<size_t>(ack.size(), sizeof a));   // 28 bytes: + the server's expert range
        if (!a[0] || (int) a[1] > r.lo || (int) a[2] < r.hi || (int) a[5] > r.elo || (int) a[6] < r.ehi) {
            err = "expert server " + r.host + " serves layers " + std::to_string(a[1]) + "-" + std::to_string(a[2]) +
                  " experts " + std::to_string(a[5]) + "-" + std::to_string(a[6]) + ", not " + std::to_string(r.lo) + "-" +
                  std::to_string(r.hi) + " experts " + std::to_string(r.elo) + "-" + std::to_string(r.ehi);
            sock_close(s); return false;
        }
        r.fd = (intptr_t) s;
        std::fprintf(stderr, "strata net: expert server %s:%d serves layers %d-%d experts %d-%d for this conductor\n",
                     r.host.c_str(), r.port, r.lo, r.hi, r.elo, r.ehi);
        routes_.push_back(r);
    }
    if (routes_.empty()) { err = "--remote-experts is empty"; return false; }
    inflight_.resize(routes_.size());
    return true;
}

NetExperts::~NetExperts() {
    for (Route& r : routes_) {
        if (r.fd >= 0) { send_frame((sock_t) r.fd, kBye, nullptr, 0); sock_close((sock_t) r.fd); }
        if (r.calls > 0)
            std::fprintf(stderr, "strata net: %s:%d layers %d-%d experts %d-%d: %llu calls, mean round trip %.3f ms (server compute %.3f ms, "
                                 "max rtt %.3f ms), %.1f KB out / %.1f KB in per call\n",
                         r.host.c_str(), r.port, r.lo, r.hi, r.elo, r.ehi, (unsigned long long) r.calls, r.rtt_ms / (double) r.calls,
                         r.server_ms / (double) r.calls, r.rtt_max_ms, (double) r.bytes_out / (double) r.calls / 1024.0,
                         (double) r.bytes_in / (double) r.calls / 1024.0);
        if (r.p_calls > 0)
            std::fprintf(stderr, "strata net: %s:%d layers %d-%d, prompt path: %llu windows, %llu rows, %.1f ms (server compute %.1f ms), "
                                 "%.1f MB out / %.1f MB in\n",
                         r.host.c_str(), r.port, r.lo, r.hi, (unsigned long long) r.p_calls, (unsigned long long) r.p_rows, r.p_ms,
                         r.p_server_ms, (double) r.p_bytes_out / 1e6, (double) r.p_bytes_in / 1e6);
    }
    if (layer_calls_ > 0)
        std::fprintf(stderr, "strata net: decode: %llu layer calls, %llu with 2+ hosts in flight at once, mean wait after the local CPU "
                             "work %.3f ms\n", (unsigned long long) layer_calls_, (unsigned long long) multi_calls_,
                     wait_ms_ / (double) layer_calls_);
}

bool NetExperts::send_req(Route& r, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                          const uint8_t* mask, int arith, uint64_t& seq, std::vector<uint8_t>& buf, std::string& err) {
    const int64_t n = n_tok * k;
    const size_t mask_len = (size_t) ((n + 3) / 4 * 4);
    buf.clear();
    put<uint32_t>(buf, (uint32_t) layer); put<uint32_t>(buf, (uint32_t) n_tok); put<uint32_t>(buf, (uint32_t) k);
    put<uint32_t>(buf, (uint32_t) arith); seq = ++seq_; put<uint64_t>(buf, seq);
    const size_t at = buf.size();
    buf.resize(at + (size_t) n * 4 + mask_len + (size_t) n_tok * H * 4, 0);
    std::memcpy(buf.data() + at, ids, (size_t) n * 4);
    for (int64_t i = 0; i < n; ++i) buf[at + (size_t) n * 4 + (size_t) i] = mask[i] ? 1 : 0;
    std::memcpy(buf.data() + at + (size_t) n * 4 + mask_len, x, (size_t) n_tok * H * 4);
    if (!send_frame((sock_t) r.fd, kReq, buf.data(), buf.size())) {
        err = "expert server " + r.host + ": send failed (layer " + std::to_string(layer) + ")";
        return false;
    }
    return true;
}

bool NetExperts::recv_resp(Route& r, uint64_t seq, uint32_t want, std::vector<uint8_t>& resp, double& server_us,
                           std::string& err) {
    uint32_t type = 0;
    if (!recv_frame((sock_t) r.fd, type, resp)) { err = "expert server " + r.host + ": connection lost"; return false; }
    if (type == kErr) { err = "expert server " + r.host + ": " + std::string(resp.begin(), resp.end()); return false; }
    uint64_t rseq = 0; uint32_t rows = 0;
    server_us = 0;
    if (type == kResp && resp.size() >= 24) {
        std::memcpy(&rseq, resp.data(), 8); std::memcpy(&server_us, resp.data() + 8, 8); std::memcpy(&rows, resp.data() + 16, 4);
    }
    if (type != kResp || rseq != seq || rows != want || resp.size() != 24 + (size_t) rows * H * 4) {
        err = "expert server " + r.host + ": response does not match the request"; return false;
    }
    return true;
}

bool NetExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
                       int arith, std::string& err) {
    const int64_t n = n_tok * k;
    any_inflight_ = false;
    int sent = 0;
    t_send_ms_ = now_ms();
    for (size_t ri = 0; ri < routes_.size(); ++ri) {
        Route& r = routes_[ri];
        InFlight& f = inflight_[ri];
        f.active = false;
        if (layer < r.lo || layer > r.hi) continue;
        f.mask.assign((size_t) n, 0);
        f.owned.clear();
        for (int64_t i = 0; i < n; ++i)
            if (kind[i] == -1 && ids[i] >= r.elo && ids[i] <= r.ehi) { f.mask[(size_t) i] = 1; f.owned.push_back((int32_t) i); }
        if (f.owned.empty()) continue;
        if (!send_req(r, layer, x, ids, n_tok, k, f.mask.data(), arith, f.seq, f.buf, err)) return false;
        r.bytes_out += f.buf.size() + sizeof(Hdr);
        f.t_send = now_ms();
        f.active = true;
        ++sent;
    }
    for (const InFlight& f : inflight_)   // staged by finish() into these rows
        if (f.active) for (int32_t i : f.owned) kind[i] = 3;
    if (sent > 0) { any_inflight_ = true; ++layer_calls_; if (sent > 1) ++multi_calls_; }
    return true;
}

bool NetExperts::finish(float* out, std::string& err) {
    if (!any_inflight_) return true;
    any_inflight_ = false;
    const double tw = now_ms();
    for (size_t ri = 0; ri < routes_.size(); ++ri) {
        InFlight& f = inflight_[ri];
        if (!f.active) continue;
        f.active = false;
        Route& r = routes_[ri];
        double server_us = 0;
        if (!recv_resp(r, f.seq, (uint32_t) f.owned.size(), resp_, server_us, err)) return false;
        const float* src = (const float*) (resp_.data() + 24);
        for (size_t q = 0; q < f.owned.size(); ++q)
            std::memcpy(out + (size_t) f.owned[q] * H, src + q * (size_t) H, (size_t) H * 4);
        const double rtt = now_ms() - f.t_send;
        r.rtt_ms += rtt; r.server_ms += server_us / 1000.0; ++r.calls; r.bytes_in += resp_.size() + sizeof(Hdr);
        if (rtt > r.rtt_max_ms) r.rtt_max_ms = rtt;
    }
    wait_ms_ += now_ms() - tw;
    return true;
}

bool NetExperts::prompt(int64_t layer, const float* x, const int32_t* ids, int64_t T, int64_t k, const uint8_t* mask,
                        float* const* dst, std::string& err) {
    constexpr int64_t W = strata::kernels::kWindowMaxT;
    static const int depth = [] { const char* v = std::getenv("STRATA_NET_PIPELINE"); return v ? std::max(1, std::atoi(v)) : 4; }();
    struct Req { size_t route; int64_t t0w, nt; uint64_t seq; uint32_t want; };
    std::vector<std::vector<Req>> q(routes_.size());   // per host, in send order (one socket each: answers come in order)
    std::vector<size_t> head(routes_.size(), 0);
    std::vector<uint8_t> m, buf;
    const double t0 = now_ms();
    bool any = false;
    auto drain_one = [&](size_t ri) -> bool {   // the oldest answer of host ri
        Req& rq = q[ri][head[ri]++];
        Route& r = routes_[ri];
        double server_us = 0;
        if (!recv_resp(r, rq.seq, rq.want, resp_, server_us, err)) return false;
        const float* src = (const float*) (resp_.data() + 24);
        size_t qq = 0;
        const int64_t n = rq.nt * k;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = rq.t0w * k + i;
            const int32_t id = ids[e];
            if (mask[e] && id >= r.elo && id <= r.ehi) std::memcpy(dst[(size_t) e], src + (qq++) * (size_t) H, (size_t) H * 4);
        }
        ++r.p_calls; r.p_rows += rq.want; r.p_server_ms += server_us / 1000.0; r.p_bytes_in += resp_.size() + sizeof(Hdr);
        return true;
    };
    for (int64_t t0w = 0; t0w < T; t0w += W) {
        const int64_t nt = std::min<int64_t>(W, T - t0w), n = nt * k;
        for (size_t ri = 0; ri < routes_.size(); ++ri) {
            Route& r = routes_[ri];
            if (layer < r.lo || layer > r.hi) continue;
            m.assign((size_t) n, 0);
            uint32_t want = 0;
            for (int64_t i = 0; i < n; ++i) {
                const int32_t id = ids[t0w * k + i];
                if (mask[t0w * k + i] && id >= r.elo && id <= r.ehi) { m[(size_t) i] = 1; ++want; }
            }
            if (want == 0) continue;
            while (q[ri].size() - head[ri] >= (size_t) depth)   // bounded: neither side's socket buffers fill up
                if (!drain_one(ri)) return false;
            uint64_t seq = 0;
            if (!send_req(r, layer, x + (size_t) t0w * H, ids + t0w * k, nt, k, m.data(), 0, seq, buf, err)) return false;
            r.p_bytes_out += buf.size() + sizeof(Hdr);
            q[ri].push_back(Req{ri, t0w, nt, seq, want});
            any = true;
        }
    }
    for (size_t ri = 0; ri < routes_.size(); ++ri)
        while (head[ri] < q[ri].size())
            if (!drain_one(ri)) return false;
    if (any) {
        const double ms = now_ms() - t0;
        for (size_t ri = 0; ri < routes_.size(); ++ri) if (!q[ri].empty()) routes_[ri].p_ms += ms;
    }
    // every masked entry must have had an owner
    for (int64_t e = 0; e < T * k; ++e) {
        if (!mask[e]) continue;
        bool owned = false;
        for (const Route& r : routes_)
            if (layer >= r.lo && layer <= r.hi && ids[e] >= r.elo && ids[e] <= r.ehi) { owned = true; break; }
        if (!owned) { err = "no expert server owns expert " + std::to_string(ids[e]) + " of layer " + std::to_string(layer); return false; }
    }
    return true;
}

// ================================================================== server
int serve_experts(ExpertDispatch& d, const ExpertServeCfg& cfg) {
    sock_init();
    const int lo = cfg.layer_lo, hi = cfg.layer_hi < 0 ? (int) cfg.n_layers - 1 : cfg.layer_hi;
    const int elo = cfg.expert_lo, ehi = cfg.expert_hi < 0 ? (int) cfg.n_expert - 1 : cfg.expert_hi;
    if (cfg.warm != nullptr) cfg.warm(cfg.warm_user, lo, hi);
    sock_t ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls == kBad) { std::fprintf(stderr, "strata net: socket failed\n"); return 1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((unsigned short) cfg.port);
    if (::bind(ls, (sockaddr*) &a, sizeof a) != 0 || ::listen(ls, 4) != 0) {
        std::fprintf(stderr, "strata net: cannot listen on port %d (errno %d)\n", cfg.port, sock_errno());
        sock_close(ls);
        return 1;
    }
    std::fprintf(stderr, "strata net: expert server listening on port %d, layers %d-%d, experts %d-%d of %lld\n", cfg.port, lo, hi,
                 elo, ehi, (long long) cfg.n_expert);
    std::vector<uint8_t> payload;
    std::vector<float> out;
    std::vector<uint8_t> resp;
    for (;;) {
        sock_t s = ::accept(ls, nullptr, nullptr);
        if (s == kBad) continue;
        sock_tune(s);
        std::fprintf(stderr, "strata net: conductor connected\n");
        uint64_t calls = 0, rows_asked = 0;
        double compute_ms = 0;
        const uint64_t gpu0 = cfg.gpu_rows != nullptr ? cfg.gpu_rows(cfg.gpu_user) : 0;
        bool hello = false;
        for (;;) {
            uint32_t type = 0;
            if (!recv_frame(s, type, payload)) break;
            if (type == kBye) break;
            if (type == kHello) {
                uint32_t h[4] = {};
                if (payload.size() >= sizeof h) std::memcpy(h, payload.data(), sizeof h);
                std::vector<uint8_t> ack;
                const bool ok = payload.size() >= sizeof h && (int64_t) h[1] == H && (int64_t) h[2] == cfg.n_expert &&
                                (int64_t) h[3] == cfg.n_layers;
                if (!ok) {
                    const char* m = "model geometry differs (n_embd / n_expert / n_layers)";
                    send_frame(s, kErr, m, std::strlen(m));
                    break;
                }
                put<uint32_t>(ack, 1); put<uint32_t>(ack, (uint32_t) lo); put<uint32_t>(ack, (uint32_t) hi);
                put<uint32_t>(ack, (uint32_t) H); put<uint32_t>(ack, (uint32_t) cfg.n_expert);
                put<uint32_t>(ack, (uint32_t) elo); put<uint32_t>(ack, (uint32_t) ehi);
                send_frame(s, kHelloAck, ack.data(), ack.size());
                hello = true;
                continue;
            }
            if (type != kReq || !hello || payload.size() < 24) {
                const char* m = "unexpected frame";
                send_frame(s, kErr, m, std::strlen(m));
                break;
            }
            uint32_t layer, n_tok, k, arith; uint64_t seq;
            std::memcpy(&layer, payload.data(), 4); std::memcpy(&n_tok, payload.data() + 4, 4);
            std::memcpy(&k, payload.data() + 8, 4); std::memcpy(&arith, payload.data() + 12, 4);
            std::memcpy(&seq, payload.data() + 16, 8);
            const size_t n = (size_t) n_tok * k;
            const size_t mask_len = (n + 3) / 4 * 4;
            if ((int) layer < lo || (int) layer > hi || n_tok < 1 || n_tok > (uint32_t) strata::kernels::kWindowMaxT || k < 1 ||
                payload.size() != 24 + n * 4 + mask_len + (size_t) n_tok * H * 4) {
                const std::string m = "request outside the served layers or malformed (layer " + std::to_string(layer) + ")";
                send_frame(s, kErr, m.data(), m.size());
                break;
            }
            const int32_t* ids = (const int32_t*) (payload.data() + 24);
            const uint8_t* mask = payload.data() + 24 + n * 4;
            const float* x = (const float*) (payload.data() + 24 + n * 4 + mask_len);
            bool foreign = false;   // an asked-for expert outside --serve-experts: this host does not hold it
            for (size_t i = 0; i < n && !foreign; ++i) foreign = mask[i] && (ids[i] < elo || ids[i] > ehi);
            if (foreign) {
                const std::string m = "request for an expert outside the served experts " + std::to_string(elo) + "-" +
                                      std::to_string(ehi) + " (layer " + std::to_string(layer) + ")";
                send_frame(s, kErr, m.data(), m.size());
                break;
            }
            out.assign(n * (size_t) H, 0.0f);
            const double t0 = now_ms();
            d.layers = layer;
            d.only_mask = mask;
            strata::core::expert_pool_dispatch_multi(d, x, ids, n_tok, k, out.data(),
                                                      arith == 1 ? strata::kernels::cpu::NativeArithmetic::WindowInvariant
                                                                 : strata::kernels::cpu::NativeArithmetic::Default);
            d.only_mask = nullptr;
            if (d.failed) {
                const std::string m = std::string("expert pool failed: ") + (d.fail != nullptr ? d.fail : "?") +
                                      " (layer " + std::to_string(layer) + ")";
                send_frame(s, kErr, m.data(), m.size());
                break;
            }
            const double us = (now_ms() - t0) * 1000.0;
            uint32_t rows = 0;
            for (size_t i = 0; i < n; ++i) rows += mask[i] ? 1u : 0u;
            resp.clear();
            put<uint64_t>(resp, seq); put<double>(resp, us); put<uint32_t>(resp, rows); put<uint32_t>(resp, 0);
            const size_t at = resp.size();
            resp.resize(at + (size_t) rows * H * 4);
            size_t q = 0;
            for (size_t i = 0; i < n; ++i)
                if (mask[i]) { std::memcpy(resp.data() + at + q * (size_t) H * 4, out.data() + i * (size_t) H, (size_t) H * 4); ++q; }
            if (!send_frame(s, kResp, resp.data(), resp.size())) break;
            ++calls; compute_ms += us / 1000.0; rows_asked += rows;
        }
        sock_close(s);
        const uint64_t on_gpu = cfg.gpu_rows != nullptr ? cfg.gpu_rows(cfg.gpu_user) - gpu0 : 0;
        std::fprintf(stderr, "strata net: conductor disconnected after %llu calls, mean compute %.3f ms, %llu rows asked, %llu on GPU (%.1f%%)\n",
                     (unsigned long long) calls, calls ? compute_ms / (double) calls : 0.0, (unsigned long long) rows_asked,
                     (unsigned long long) on_gpu, rows_asked ? 100.0 * (double) on_gpu / (double) rows_asked : 0.0);
        if (cfg.once) break;
    }
    sock_close(ls);
    return 0;
}

}  // namespace strata::core
