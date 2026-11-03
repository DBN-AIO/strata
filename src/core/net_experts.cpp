// src/core/net_experts.cpp - remote experts over the LAN: client (conductor side) and expert server.
// Protocol: include/strata/core/net_experts.hpp.  Sockets: the small shim below is the only OS specific code.
// Threat model and flags: docs/REMOTE_EXPERTS.md ("Security").
#include "strata/core/net_experts.hpp"

#include "strata/core/net_proto.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
#endif

namespace strata::core {
namespace {
using Clock = std::chrono::steady_clock;
namespace np = netproto;
constexpr int64_t kFlashEmbd = strata::kernels::cpu::H;   // the default n_embd of a peer that does not say (Flash-Next)
constexpr uint32_t kMagic = 0x31505853u;   // "SXP1"
constexpr uint32_t kProtoVersion = 2;
enum : uint32_t { kHello = 1, kHelloAck = 2, kReq = 3, kResp = 4, kErr = 5, kBye = 6, kChallenge = 7 };
// frame size caps: a peer that has not authenticated may make us allocate 4 KiB, never more.  The largest legitimate REQ is
// 24 + 256*4 + 256 + 16*H*4 bytes (~165 KB), the largest RESP 24 + 256*H*4 (~2.6 MB).
constexpr uint64_t kMaxHandshakeFrame = 4096;
constexpr uint64_t kMaxReqFrame = 1ull << 20;
constexpr uint64_t kMaxRespFrame = 4ull << 20;
constexpr int kMaxHandshakes = 8;          // concurrent unauthenticated connections the server will hold
constexpr int kDefaultTimeoutMs = 10000;

double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- wire structs (no padding; sizes asserted)
struct Hdr { uint32_t magic, type; uint64_t len; };
struct ChallengeMsg { uint32_t version, flags; uint8_t nonce[32]; };
struct HelloMsg { uint32_t version, n_embd, n_expert, n_layers, isa, pad; uint8_t fp[32]; uint8_t mac[32]; };
struct AckMsg { uint32_t version, ok, lo, hi, n_embd, n_expert, top_k, isa; };
static_assert(sizeof(Hdr) == 16, "SXP frame header is 16 bytes");
static_assert(sizeof(ChallengeMsg) == 40, "SXP CHALLENGE is 40 bytes");
static_assert(sizeof(HelloMsg) == 88, "SXP HELLO is 88 bytes");
static_assert(sizeof(AckMsg) == 32, "SXP HELLO_ACK is 32 bytes");
constexpr size_t kHelloSigned = offsetof(HelloMsg, mac);

// ---------------------------------------------------------------- socket shim (POSIX / Winsock)
#ifdef _WIN32
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
bool sock_init() {
    static const bool ok = [] {   // function-local static: initialised once, thread safe
        WSADATA w;
        const int r = WSAStartup(MAKEWORD(2, 2), &w);
        if (r != 0) std::fprintf(stderr, "strata net: WSAStartup failed (%d)\n", r);
        return r == 0;
    }();
    return ok;
}
void sock_close(sock_t s) { if (s != kBad) closesocket(s); }
int sock_errno() { return WSAGetLastError(); }
bool err_timeout(int e) { return e == WSAETIMEDOUT; }
bool err_intr(int e) { return e == WSAEINTR; }
bool err_inprogress(int e) { return e == WSAEWOULDBLOCK; }
bool set_nonblock(sock_t s, bool nb) { u_long v = nb ? 1 : 0; return ioctlsocket(s, FIONBIO, &v) == 0; }
int sock_poll(sock_t s, bool want_write, int ms) {
    WSAPOLLFD p{}; p.fd = s; p.events = want_write ? POLLWRNORMAL : POLLRDNORMAL;
    return WSAPoll(&p, 1, ms);
}
void set_io_timeouts(sock_t s, int rcv_ms, int snd_ms) {
    DWORD r = (DWORD) rcv_ms, w = (DWORD) snd_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &r, sizeof r);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*) &w, sizeof w);
}
#else
using sock_t = int;
constexpr sock_t kBad = -1;
bool sock_init() { return true; }
void sock_close(sock_t s) { if (s != kBad) ::close(s); }
int sock_errno() { return errno; }
bool err_timeout(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
bool err_intr(int e) { return e == EINTR; }
bool err_inprogress(int e) { return e == EINPROGRESS; }
bool set_nonblock(sock_t s, bool nb) {
    const int f = fcntl(s, F_GETFL, 0);
    return f >= 0 && fcntl(s, F_SETFL, nb ? (f | O_NONBLOCK) : (f & ~O_NONBLOCK)) == 0;
}
int sock_poll(sock_t s, bool want_write, int ms) {
    pollfd p{}; p.fd = s; p.events = want_write ? POLLOUT : POLLIN;
    for (;;) {
        const int r = ::poll(&p, 1, ms);
        if (r < 0 && errno == EINTR) continue;   // a signal: ask again (the wait restarts; the callers' waits are short)
        return r;
    }
}
void set_io_timeouts(sock_t s, int rcv_ms, int snd_ms) {   // 0: no timeout
    timeval r{rcv_ms / 1000, (rcv_ms % 1000) * 1000}, w{snd_ms / 1000, (snd_ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &r, sizeof r);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*) &w, sizeof w);
}
#endif

void ignore_sigpipe() {
#ifndef _WIN32
    static std::once_flag once;
    std::call_once(once, [] { std::signal(SIGPIPE, SIG_IGN); });   // a peer that died mid-frame must not kill this process
#endif
}

std::atomic<int> g_timeout_ms{0};
int timeout_ms() {
    int v = g_timeout_ms.load(std::memory_order_relaxed);
    if (v > 0) return v;
    v = kDefaultTimeoutMs;
    if (const char* e = std::getenv("STRATA_EXPERT_TIMEOUT_MS")) {
        long x = 0;
        if (np::parse_uint(e, 100, 600000, x)) v = (int) x;
        else std::fprintf(stderr, "strata net: STRATA_EXPERT_TIMEOUT_MS must be 100..600000; using %d\n", v);
    }
    g_timeout_ms.store(v, std::memory_order_relaxed);
    return v;
}

std::string env_token() {
    const char* t = std::getenv("STRATA_EXPERT_TOKEN");
    return t != nullptr ? std::string(t) : std::string();
}

void sock_tune(sock_t s, bool keepalive) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 4 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
#ifdef SO_NOSIGPIPE
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, (const char*) &one, sizeof one);   // macOS: no SIGPIPE on a closed peer
#endif
    if (keepalive) {   // the server's idle session waits without a read timeout: keepalive notices a dead host
        setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (const char*) &one, sizeof one);
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
        int idle = 30, intvl = 10, cnt = 3;
        setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, (const char*) &idle, sizeof idle);
        setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, (const char*) &intvl, sizeof intvl);
        setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, (const char*) &cnt, sizeof cnt);
#endif
    }
}

// why the last send/recv on this thread failed (for the log line the caller prints)
thread_local std::string g_why;
void why_errno(int e) {
    g_why = err_timeout(e) ? "timed out waiting for the peer" : "socket error (errno " + std::to_string(e) + ")";
}

bool send_all(sock_t s, const void* p, size_t n) {
    const char* c = (const char*) p;
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;   // Linux: EPIPE instead of SIGPIPE
#else
    constexpr int flags = 0;
#endif
    while (n > 0) {
        const int r = (int) ::send(s, c, (int) (n > (1u << 30) ? (1u << 30) : n), flags);
        if (r > 0) { c += r; n -= (size_t) r; continue; }
        const int e = sock_errno();
        if (r < 0 && err_intr(e)) continue;
        if (r == 0) g_why = "connection closed by the peer"; else why_errno(e);
        return false;
    }
    return true;
}
/// `deadline_ms` > 0: an absolute steady-clock time after which the read gives up (a peer trickling bytes cannot hold a
/// handshake open past it).
bool recv_all(sock_t s, void* p, size_t n, double deadline_ms = 0) {
    char* c = (char*) p;
    while (n > 0) {
        if (deadline_ms > 0 && now_ms() > deadline_ms) { g_why = "timed out waiting for the peer"; return false; }
        const int r = (int) ::recv(s, c, (int) (n > (1u << 30) ? (1u << 30) : n), 0);
        if (r > 0) { c += r; n -= (size_t) r; continue; }
        const int e = sock_errno();
        if (r < 0 && err_intr(e)) continue;
        if (r == 0) g_why = "connection closed by the peer"; else why_errno(e);
        return false;
    }
    return true;
}

bool send_frame(sock_t s, uint32_t type, const void* payload, uint64_t len) {
    Hdr h{kMagic, type, len};
    if (!send_all(s, &h, sizeof h)) return false;
    return len == 0 || send_all(s, payload, (size_t) len);
}
bool recv_frame(sock_t s, uint32_t& type, std::vector<uint8_t>& payload, uint64_t max_len, double deadline_ms = 0) {
    Hdr h;
    if (!recv_all(s, &h, sizeof h, deadline_ms)) return false;
    if (h.magic != kMagic) { g_why = "not an SXP frame (bad magic)"; return false; }
    if (h.len > max_len) { g_why = "frame of " + std::to_string(h.len) + " bytes exceeds the limit " + std::to_string(max_len); return false; }
    type = h.type;
    payload.resize((size_t) h.len);
    return h.len == 0 || recv_all(s, payload.data(), (size_t) h.len, deadline_ms);
}
void send_err(sock_t s, const std::string& m) { send_frame(s, kErr, m.data(), m.size()); }

std::string peer_str(sock_t s) {
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    char host[NI_MAXHOST] = "?";
    if (getpeername(s, (sockaddr*) &ss, &sl) == 0) getnameinfo((sockaddr*) &ss, sl, host, sizeof host, nullptr, 0, NI_NUMERICHOST);
    return host;
}

/// connect with a timeout (non-blocking connect + poll), then blocking I/O with send/recv timeouts.
sock_t sock_connect(const std::string& host, int port, int tmo_ms, std::string& err) {
    if (!sock_init()) { err = "Winsock did not start"; return kBad; }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return kBad;
    }
    sock_t s = kBad;
    int last_errno = 0;
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == kBad) continue;
        bool ok = false;
        if (set_nonblock(s, true)) {
            if (::connect(s, a->ai_addr, (int) a->ai_addrlen) == 0) ok = true;
            else if (err_inprogress(sock_errno()) && sock_poll(s, true, tmo_ms) > 0) {
                int so = 0;
                socklen_t sl = sizeof so;
                ok = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &so, &sl) == 0 && so == 0;
                if (!ok) last_errno = so;
            } else last_errno = sock_errno();
        }
        if (ok && set_nonblock(s, false)) break;
        sock_close(s);
        s = kBad;
    }
    freeaddrinfo(res);
    if (s == kBad) {
        err = "cannot connect to " + host + ":" + ps + " within " + std::to_string(tmo_ms) + " ms (errno " + std::to_string(last_errno) + ")";
        return kBad;
    }
    sock_tune(s, false);
    set_io_timeouts(s, tmo_ms, tmo_ms);
    return s;
}

template <class T> void put(std::vector<uint8_t>& b, const T& v) {
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
}

void hello_mac(const std::string& token, const uint8_t nonce[32], const HelloMsg& h, uint8_t out[32]) {
    np::hmac_sha256(token.data(), token.size(), nonce, 32, &h, kHelloSigned, out);
}

void random_bytes(uint8_t* p, size_t n) {
    std::random_device rd;   // getrandom / /dev/urandom / RtlGenRandom
    for (size_t i = 0; i < n; i += 4) {
        uint32_t v = rd() ^ (uint32_t) Clock::now().time_since_epoch().count();
        std::memcpy(p + i, &v, std::min<size_t>(4, n - i));
    }
}
}  // namespace

// ================================================================== identity
void net_set_timeout_ms(int ms) { if (ms >= 100) g_timeout_ms.store(ms, std::memory_order_relaxed); }

std::string net_isa_name(uint32_t isa) {
    std::string s = (isa & 1) ? "avx512" : (isa & 2) ? "avx512bw" : (isa & 4) ? "avx2" : "baseline";
    const uint32_t cap = isa >> 8;
    if (cap != 3 && cap != 0) s += " (isa cap " + std::to_string(cap) + ")";
    return s;
}

NetIdentity net_identity(const std::vector<std::string>& files) {
    using namespace strata::kernels::cpu;
    const ExpertLayout& lay = expert_layout();
    np::Sha256 h;
    auto u64 = [&](uint64_t v) { h.update(&v, 8); };
    h.update("SXPFP2", 6);
    u64(lay.native); u64((uint64_t) lay.n_layers); u64((uint64_t) lay.n_expert); u64(lay.total); u64((uint64_t) lay.version); u64(lay.max_blob);
    u64(lay.fmt.size());
    for (const NativeFmt& f : lay.fmt) {   // the expert tensor types and geometry, field by field (no struct padding in the hash)
        u64((uint64_t) f.gu_type); u64((uint64_t) f.d_type); u64((uint64_t) f.gu_act); u64((uint64_t) f.d_act);
        u64((uint64_t) f.n_embd); u64((uint64_t) f.n_ff); u64(f.gu_row); u64(f.d_row); u64(f.up_off); u64(f.down_off);
        u64(f.bytes); u64(f.tail_off); u64((uint64_t) f.had_block); u64(f.had_gu); u64(f.had_d);
    }
    u64(lay.offset.size()); for (uint64_t v : lay.offset) u64(v);
    u64(lay.bytes.size()); for (uint64_t v : lay.bytes) u64(v);
    u64(lay.gguf_off.size()); for (uint64_t v : lay.gguf_off) u64(v);
    u64(lay.gguf_file.size());
    for (const std::string& f : lay.gguf_file) { u64(f.size()); h.update(f.data(), f.size()); }
    std::vector<std::string> seen;   // the files' sizes (not their paths: the hosts keep them elsewhere)
    for (const std::string& f : files) {
        if (std::find(seen.begin(), seen.end(), f) != seen.end()) continue;
        seen.push_back(f);
        std::error_code ec;
        const uint64_t sz = (uint64_t) std::filesystem::file_size(f, ec);
        u64(ec ? ~0ull : sz);
    }
    u64(seen.size());
    NetIdentity id;
    h.finish(id.fingerprint);
    id.isa = net_isa_bits();
    return id;
}

uint32_t net_isa_bits() {
    using namespace strata::kernels::cpu;
    return (cpu_avx512_ok() ? 1u : 0u) | (cpu_avx512bw_ok() ? 2u : 0u) | (cpu_avx2_ok() ? 4u : 0u) | ((uint32_t) cpu_isa_cap() << 8);
}

// ================================================================== client
// B5 (request pipelining across hosts): several hosts may serve one layer, each owning a range of its experts
// (host:port=lo-hi/e0-e1).  begin() sends the layer's request to every owning host before the conductor's own CPU work and
// finish() collects all answers, so the hosts compute their shares of the same layer at the same time and the layer waits
// for the slowest one instead of one host doing all of it.  The prompt path keeps up to STRATA_NET_PIPELINE windows
// (default 4) in flight per host instead of one round trip per window.
bool NetExperts::connect_route(Route& r, int tmo_ms, std::string& err) {
    const int64_t H = n_embd_;
    const sock_t s = sock_connect(r.host, r.port, tmo_ms, err);
    if (s == kBad) return false;
    auto fail = [&](const std::string& m) { err = m; sock_close(s); return false; };
    uint32_t type = 0;
    std::vector<uint8_t> in;
    if (!recv_frame(s, type, in, kMaxHandshakeFrame, now_ms() + tmo_ms)) return fail("no challenge from expert server " + r.host + " (" + g_why + ")");
    if (type == kErr) return fail("expert server " + r.host + ": " + std::string(in.begin(), in.end()));
    ChallengeMsg ch{};
    if (type != kChallenge || in.size() != sizeof ch) return fail("expert server " + r.host + " did not send a challenge (is it a strata expert server of this version?)");
    std::memcpy(&ch, in.data(), sizeof ch);
    if (ch.version != kProtoVersion)
        return fail("expert server " + r.host + " speaks protocol version " + std::to_string(ch.version) + ", this build speaks " + std::to_string(kProtoVersion));
    const bool need_token = (ch.flags & 1u) != 0;
    if (need_token && token_.empty()) return fail("expert server " + r.host + " requires a token: set STRATA_EXPERT_TOKEN");
    HelloMsg h{};
    h.version = kProtoVersion; h.n_embd = (uint32_t) H; h.n_expert = (uint32_t) n_expert_; h.n_layers = (uint32_t) n_layers_;
    h.isa = id_.isa;
    std::memcpy(h.fp, id_.fingerprint, 32);
    if (!token_.empty()) hello_mac(token_, ch.nonce, h, h.mac);
    if (!send_frame(s, kHello, &h, sizeof h)) return fail("expert server " + r.host + ": HELLO not sent (" + g_why + ")");
    if (!recv_frame(s, type, in, kMaxHandshakeFrame, now_ms() + tmo_ms)) return fail("no handshake answer from " + r.host + " (" + g_why + ")");
    if (type == kErr) return fail("expert server " + r.host + ": " + std::string(in.begin(), in.end()));
    AckMsg a{};
    if (type != kHelloAck || in.size() != sizeof a) return fail("bad handshake from " + r.host);
    std::memcpy(&a, in.data(), sizeof a);
    if (a.version != kProtoVersion) return fail("expert server " + r.host + " answered with protocol version " + std::to_string(a.version));
    if (a.ok != 1 || (int) a.lo > r.lo || (int) a.hi < r.hi)
        return fail("expert server " + r.host + " serves layers " + std::to_string(a.lo) + "-" + std::to_string(a.hi) + ", not " +
                    std::to_string(r.lo) + "-" + std::to_string(r.hi));
    if (a.n_embd != (uint32_t) H || (int64_t) a.n_expert != n_expert_) return fail("expert server " + r.host + ": model geometry differs");
    if (a.isa != id_.isa)   // the server refuses a mismatch itself; this is the defence if it does not
        return fail("expert server " + r.host + " runs the " + net_isa_name(a.isa) + " kernels, this host the " + net_isa_name(id_.isa) + " ones");
    if (!need_token && !token_.empty())
        std::fprintf(stderr, "strata net: expert server %s does not ask for a token although STRATA_EXPERT_TOKEN is set here\n", r.host.c_str());
    r.top_k = (int) a.top_k;
    r.fd = (intptr_t) s;
    return true;
}

bool NetExperts::open(const std::string& spec, int64_t n_layers, int64_t n_expert, const NetIdentity& id, std::string& err, int64_t n_embd) {
    ignore_sigpipe();
    n_embd_ = n_embd > 0 ? n_embd : kFlashEmbd;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    id_ = id;
    token_ = env_token();
    size_t pos = 0;
    std::vector<np::RouteSpec> specs;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string item = spec.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty()) continue;
        np::RouteSpec rs;
        std::string why;
        if (!np::parse_route_item(item, n_layers, n_expert, rs, why)) { err = "--remote-experts " + why; return false; }
        for (const np::RouteSpec& o : specs)   // two hosts may share layers only with disjoint expert ranges
            if (np::routes_conflict(rs, o)) { err = "overlapping layer and expert ranges in --remote-experts: " + item; return false; }
        specs.push_back(rs);
        Route r;
        r.host = rs.host; r.port = rs.port; r.lo = rs.lo; r.hi = rs.hi; r.elo = rs.elo; r.ehi = rs.ehi;
        if (!connect_route(r, timeout_ms(), err)) return false;
        std::fprintf(stderr, "strata net: expert server %s:%d serves layers %d-%d experts %d-%d for this conductor (auth %s, %s kernels, timeout %d ms)\n",
                     r.host.c_str(), r.port, r.lo, r.hi, r.elo, r.ehi, token_.empty() ? "off" : "token", net_isa_name(id_.isa).c_str(), timeout_ms());
        routes_.push_back(r);
    }
    if (routes_.empty()) { err = "--remote-experts is empty"; return false; }
    inflight_.assign(routes_.size(), InFlight());
    return true;
}

bool NetExperts::covers(int64_t layer) const {
    std::vector<np::RouteSpec> rs;
    for (const Route& r : routes_) { np::RouteSpec s; s.lo = r.lo; s.hi = r.hi; s.elo = r.elo; s.ehi = r.ehi; rs.push_back(s); }
    return np::layer_covered(rs, layer, n_expert_);
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
        if (r.outages > 0)
            std::fprintf(stderr, "strata net: %s:%d: %llu outage(s), %llu reconnect(s), %llu expert rows computed locally instead\n",
                         r.host.c_str(), r.port, (unsigned long long) r.outages, (unsigned long long) r.reconnects, (unsigned long long) r.local_rows);
    }
    if (layer_calls_ > 0)
        std::fprintf(stderr, "strata net: decode: %llu layer calls, %llu with 2+ hosts in flight at once; mean wait for the answers after the "
                             "local CPU work %.3f ms\n", (unsigned long long) layer_calls_, (unsigned long long) multi_calls_,
                     wait_ms_ / (double) layer_calls_);
}

void NetExperts::mark_down(Route& r, const std::string& why) {
    const bool was_up = r.fd >= 0;
    if (was_up) { sock_close((sock_t) r.fd); r.fd = -1; ++r.outages; }
    r.backoff_ms = r.backoff_ms <= 0 ? 1000 : std::min(r.backoff_ms * 2, 30000);
    r.retry_at_ms = now_ms() + r.backoff_ms;
    if (was_up)   // once per outage: the failed retries that follow stay quiet
        std::fprintf(stderr, "strata net: expert server %s:%d (layers %d-%d) lost: %s; computing those experts locally, next try in %d ms\n",
                     r.host.c_str(), r.port, r.lo, r.hi, why.c_str(), r.backoff_ms);
}

void NetExperts::abandon() {
    if (cur_ < 0) return;
    cur_ = -1;
    for (size_t ri = 0; ri < routes_.size(); ++ri)
        if (inflight_[ri].active) { inflight_[ri].active = false; mark_down(routes_[ri], "the request was dropped before its answer was read"); }
}

/// A route that is down is retried once its backoff has passed (a short connect timeout: a dead host must not stall decode).
bool NetExperts::route_up(Route& r) {
    if (r.fd >= 0) return true;
    if (now_ms() < r.retry_at_ms) return false;
    std::string cerr;
    if (!connect_route(r, std::min(timeout_ms(), 2000), cerr)) { mark_down(r, cerr); return false; }
    ++r.reconnects;
    std::fprintf(stderr, "strata net: expert server %s:%d (layers %d-%d) is back\n", r.host.c_str(), r.port, r.lo, r.hi);
    return true;
}

bool NetExperts::send_req(Route& r, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const uint8_t* mask,
                          int arith, uint64_t& seq, std::vector<uint8_t>& buf) {
    const int64_t H = n_embd_;
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
    return send_frame((sock_t) r.fd, kReq, buf.data(), buf.size());
}

std::string NetExperts::recv_resp(Route& r, uint64_t seq, uint32_t want, std::vector<uint8_t>& resp, double& server_us) {
    const int64_t H = n_embd_;
    uint32_t type = 0;
    server_us = 0;
    if (!recv_frame((sock_t) r.fd, type, resp, kMaxRespFrame)) return g_why.empty() ? std::string("receive failed") : g_why;
    if (type == kErr) return "server refused the request: " + std::string(resp.begin(), resp.end());
    if (type != kResp || resp.size() < 24) return "bad response";
    uint64_t rseq; uint32_t rows;
    std::memcpy(&rseq, resp.data(), 8); std::memcpy(&server_us, resp.data() + 8, 8); std::memcpy(&rows, resp.data() + 16, 4);
    if (rseq != seq || rows != want || resp.size() != 24 + (size_t) rows * H * 4) return "response does not match the request";
    return std::string();
}

bool NetExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
                       int arith, std::string& err) {
    cur_ = -1;
    owned_.clear();
    const int64_t n = n_tok * k;
    int sent = 0;
    for (size_t ri = 0; ri < routes_.size(); ++ri) {
        Route& r = routes_[ri];
        InFlight& f = inflight_[ri];
        f.active = false;
        if (layer < r.lo || layer > r.hi) continue;
        f.mask.assign((size_t) n, 0);
        f.owned.clear();
        for (int64_t i = 0; i < n; ++i)
            if (kind[i] == -1 && ids[i] >= r.elo && ids[i] <= r.ehi && ids[i] < n_expert_) { f.mask[(size_t) i] = 1; f.owned.push_back((int32_t) i); }
        if (f.owned.empty()) continue;
        if (r.top_k > 0 && k != r.top_k) continue;   // not the model the server was started for: local
        if (!route_up(r)) { r.local_rows += f.owned.size(); continue; }
        if (!send_req(r, layer, x, ids, n_tok, k, f.mask.data(), arith, f.seq, f.buf)) {
            err = "send failed (layer " + std::to_string(layer) + "): " + g_why;
            mark_down(r, err);
            r.local_rows += f.owned.size();
            continue;   // nothing marked: the CPU work covers those rows
        }
        r.bytes_out += f.buf.size() + sizeof(Hdr);
        f.t_send = now_ms();
        f.active = true;
        ++sent;
    }
    if (sent == 0) return true;
    for (const InFlight& f : inflight_)   // staged by finish() into these rows
        if (f.active) for (int32_t i : f.owned) { kind[i] = 3; owned_.push_back(i); }
    cur_ = 0;
    ++layer_calls_;
    if (sent > 1) ++multi_calls_;
    return true;
}

bool NetExperts::finish(float* out, std::string& err) {
    const int64_t H = n_embd_;
    if (cur_ < 0) return true;
    cur_ = -1;
    owned_.clear();   // from here: the entries the caller must compute itself (those of the hosts that failed)
    const double tw = now_ms();
    bool ok = true;
    for (size_t ri = 0; ri < routes_.size(); ++ri) {
        InFlight& f = inflight_[ri];
        if (!f.active) continue;
        f.active = false;
        Route& r = routes_[ri];
        double server_us = 0;
        const std::string why = recv_resp(r, f.seq, (uint32_t) f.owned.size(), resp_, server_us);
        if (!why.empty()) {   // this host's rows are the caller's; the other hosts' answers are still read (their streams stay in step)
            err = why;
            mark_down(r, why);
            r.local_rows += f.owned.size();
            owned_.insert(owned_.end(), f.owned.begin(), f.owned.end());
            ok = false;
            continue;
        }
        const float* src = (const float*) (resp_.data() + 24);
        for (size_t q = 0; q < f.owned.size(); ++q)
            std::memcpy(out + (size_t) f.owned[q] * H, src + q * (size_t) H, (size_t) H * 4);
        const double rtt = now_ms() - f.t_send;
        r.rtt_ms += rtt; r.server_ms += server_us / 1000.0; ++r.calls; r.bytes_in += resp_.size() + sizeof(Hdr);
        if (rtt > r.rtt_max_ms) r.rtt_max_ms = rtt;
        if (r.backoff_ms != 0) r.backoff_ms = 0;   // a healthy call ends the outage's backoff
    }
    wait_ms_ += now_ms() - tw;
    std::sort(owned_.begin(), owned_.end());
    return ok;
}

bool NetExperts::prompt(int64_t layer, const float* x, const int32_t* ids, int64_t T, int64_t k, const uint8_t* mask,
                        float* const* dst, std::string& err) {
    const int64_t H = n_embd_;
    if (cur_ >= 0) { err = "a decode request is in flight"; return false; }
    static const int depth = [] { const char* v = std::getenv("STRATA_NET_PIPELINE"); return v ? std::max(1, std::atoi(v)) : 4; }();
    struct Req { int64_t t0w, nt; uint64_t seq; uint32_t want; };
    std::vector<size_t> rts;   // the hosts of this layer
    for (size_t ri = 0; ri < routes_.size(); ++ri) {
        Route& r = routes_[ri];
        if (layer < r.lo || layer > r.hi) continue;
        if (r.top_k > 0 && k != r.top_k) { err = "the server routes " + std::to_string(r.top_k) + " experts per token, not " + std::to_string(k); return false; }
        if (!route_up(r)) { err = "expert server " + r.host + " is down"; return false; }
        rts.push_back(ri);
    }
    if (rts.empty()) { err = "no expert server serves layer " + std::to_string(layer); return false; }
    std::vector<std::vector<Req>> q(routes_.size());   // per host, in send order (one socket each: its answers come in order)
    std::vector<size_t> head(routes_.size(), 0);
    std::vector<uint8_t> m;
    bool failed = false;
    auto drain_one = [&](size_t ri) -> bool {   // the oldest answer of host ri
        const Req rq = q[ri][head[ri]++];
        Route& r = routes_[ri];
        double server_us = 0;
        const std::string why = recv_resp(r, rq.seq, rq.want, resp_, server_us);
        if (!why.empty()) {
            err = "expert server " + r.host + " (prompt, layer " + std::to_string(layer) + "): " + why;
            mark_down(r, err);
            return false;
        }
        const float* src = (const float*) (resp_.data() + 24);
        size_t qq = 0;
        for (int64_t i = rq.t0w * k; i < (rq.t0w + rq.nt) * k; ++i)
            if (mask[i] && ids[i] >= r.elo && ids[i] <= r.ehi) std::memcpy(dst[(size_t) i], src + (qq++) * (size_t) H, (size_t) H * 4);
        ++r.p_calls; r.p_rows += rq.want; r.p_server_ms += server_us / 1000.0; r.p_bytes_in += resp_.size() + sizeof(Hdr);
        return true;
    };
    const double t_start = now_ms();
    // each host's share of every window (the entries asked for whose expert it owns), sent round-robin over the hosts so
    // they all work at once, each with at most `depth` windows unanswered
    std::vector<std::vector<uint8_t>> own(routes_.size());
    std::vector<std::vector<np::PromptWindow>> wins(routes_.size());
    size_t most = 0;
    for (size_t ri : rts) {
        const Route& r = routes_[ri];
        own[ri].assign((size_t) (T * k), 0);
        for (int64_t i = 0; i < T * k; ++i) own[ri][(size_t) i] = mask[i] && ids[i] >= r.elo && ids[i] <= r.ehi ? 1 : 0;
        wins[ri] = np::prompt_windows(T, k, own[ri].data(), strata::kernels::kWindowMaxT);
        most = std::max(most, wins[ri].size());
    }
    for (size_t w = 0; w < most && !failed; ++w)
        for (size_t ri : rts) {
            if (w >= wins[ri].size()) continue;
            Route& r = routes_[ri];
            const np::PromptWindow& pw = wins[ri][w];
            while (q[ri].size() - head[ri] >= (size_t) depth)   // bounded: neither side's socket buffers fill up
                if (!drain_one(ri)) { failed = true; break; }
            if (failed) break;
            uint64_t seq = 0;
            // arith 1, window-invariant: a row does not depend on its window (the local reference / fallback uses it too)
            if (!send_req(r, layer, x + (size_t) pw.t0 * H, ids + pw.t0 * k, pw.nt, k, own[ri].data() + pw.t0 * k, 1, seq, buf_)) {
                err = "expert server " + r.host + " (prompt, layer " + std::to_string(layer) + "): send failed: " + g_why;
                mark_down(r, err);
                failed = true;
                break;
            }
            r.p_bytes_out += buf_.size() + sizeof(Hdr);
            q[ri].push_back(Req{pw.t0, pw.nt, seq, pw.want});
        }
    for (size_t ri : rts)   // every answer still owed is read (or its host dropped), so no stream is left behind
        while (!failed && head[ri] < q[ri].size())
            if (!drain_one(ri)) failed = true;
    if (failed) {
        for (size_t ri : rts) if (head[ri] < q[ri].size() && routes_[ri].fd >= 0) mark_down(routes_[ri], "a prompt request failed on another host");
        return false;
    }
    const double ms = now_ms() - t_start;
    for (size_t ri : rts) { routes_[ri].p_ms += ms; if (routes_[ri].backoff_ms != 0) routes_[ri].backoff_ms = 0; }
    return true;
}

// ================================================================== server
namespace {
struct Conn {
    std::thread t;
    std::atomic<bool> done{false};
};
struct Ready { sock_t s = kBad; std::string peer; };
struct ServerCtx {
    const ExpertServeCfg* cfg = nullptr;
    int64_t H = 0;                           // n_embd of the model being served
    np::ReqLimits lim;
    std::string token;
    int tmo_ms = kDefaultTimeoutMs;
    std::atomic<bool> stop{false};
    std::atomic<bool> busy{false};           // a session owns the dispatch (one conductor at a time)
    std::atomic<int> handshaking{0};
    std::atomic<uint64_t> refused{0};        // refused connections / requests, for rate limited logging
    std::mutex mu;
    std::condition_variable cv;
    Ready ready;
};

bool log_refusal(ServerCtx& C) {   // the first 20 refusals, then every 100th: a flood must not flood the log
    const uint64_t n = ++C.refused;
    return n <= 20 || n % 100 == 0;
}

/// One connection, off the dispatch thread: challenge, authenticate, check the model, claim the single session slot.
void handshake_thread(ServerCtx* C, sock_t s) {
    const int64_t H = C->H;
    const std::string peer = peer_str(s);
    auto finish_fail = [&](const std::string& tell, const std::string& log) {
        if (!tell.empty()) send_err(s, tell);
        if (log_refusal(*C)) std::fprintf(stderr, "strata net: refused connection from %s: %s\n", peer.c_str(), log.c_str());
        sock_close(s);
        --C->handshaking;
    };
    set_io_timeouts(s, C->tmo_ms, C->tmo_ms);
    sock_tune(s, false);
    ChallengeMsg ch{};
    ch.version = kProtoVersion; ch.flags = C->token.empty() ? 0u : 1u;
    random_bytes(ch.nonce, 32);
    if (!send_frame(s, kChallenge, &ch, sizeof ch)) return finish_fail("", "challenge not sent (" + g_why + ")");
    uint32_t type = 0;
    std::vector<uint8_t> in;
    if (!recv_frame(s, type, in, kMaxHandshakeFrame, now_ms() + C->tmo_ms)) return finish_fail("", "no HELLO (" + g_why + ")");
    HelloMsg h{};
    if (type != kHello || in.size() != sizeof h) return finish_fail("expected a HELLO of this protocol version", "first frame is not a version-2 HELLO");
    std::memcpy(&h, in.data(), sizeof h);
    if (!C->token.empty()) {   // before anything else is revealed or checked
        uint8_t want[32];
        hello_mac(C->token, ch.nonce, h, want);
        if (!np::ct_equal(want, h.mac, 32)) return finish_fail("authentication failed", "authentication failed");
    }
    if (h.version != kProtoVersion)
        return finish_fail("protocol version " + std::to_string(h.version) + " is not supported (this server speaks " + std::to_string(kProtoVersion) + ")", "protocol version " + std::to_string(h.version));
    if ((int64_t) h.n_embd != H || (int64_t) h.n_expert != C->cfg->n_expert || (int64_t) h.n_layers != C->cfg->n_layers)
        return finish_fail("model geometry differs (n_embd / n_expert / n_layers)", "model geometry differs");
    if (!np::ct_equal(h.fp, C->cfg->identity.fingerprint, 32))
        return finish_fail("model fingerprint differs: server " + np::hex(C->cfg->identity.fingerprint, 32) + " (isa " + std::to_string(C->cfg->identity.isa) + "), yours " + np::hex(h.fp, 32) +
                           " (different quantization or files: the rows would not be the conductor's)", "model fingerprint differs");
    if (h.isa != C->cfg->identity.isa && std::getenv("STRATA_EXPERT_ALLOW_ISA_MISMATCH") == nullptr)
        return finish_fail("CPU kernel path differs: server " + net_isa_name(C->cfg->identity.isa) + ", yours " + net_isa_name(h.isa) +
                           " (set STRATA_EXPERT_ALLOW_ISA_MISMATCH=1 on the server to accept it: equal only within kernel tolerance)", "kernel path differs (" + net_isa_name(h.isa) + ")");
    bool expect = false;
    if (!C->busy.compare_exchange_strong(expect, true)) return finish_fail("busy: another conductor is connected", "busy (a conductor is connected)");
    AckMsg a{};
    a.version = kProtoVersion; a.ok = 1; a.lo = (uint32_t) C->cfg->layer_lo; a.hi = (uint32_t) C->cfg->layer_hi;
    a.n_embd = (uint32_t) H; a.n_expert = (uint32_t) C->cfg->n_expert; a.top_k = (uint32_t) C->cfg->top_k; a.isa = C->cfg->identity.isa;
    if (!send_frame(s, kHelloAck, &a, sizeof a)) {
        C->busy = false;
        return finish_fail("", "HELLO_ACK not sent (" + g_why + ")");
    }
    {
        std::lock_guard<std::mutex> lk(C->mu);
        C->ready.s = s;
        C->ready.peer = peer;
    }
    C->cv.notify_all();
    --C->handshaking;
}

/// The REQ loop of the one authenticated conductor, on the thread that owns the dispatch.
void serve_session(const ServeCompute& compute, ServerCtx& C, sock_t s, const std::string& peer, std::vector<uint8_t>& payload,
                   std::vector<float>& out, std::vector<uint8_t>& resp) {
    const int64_t H = C.H;
    std::fprintf(stderr, "strata net: conductor connected from %s\n", peer.c_str());
    sock_tune(s, true);
    set_io_timeouts(s, 0, C.tmo_ms);   // a conductor may sit idle between prompts (keepalive covers a dead one); a send never waits forever
    uint64_t calls = 0, bad_reqs = 0, rows_asked = 0;
    double compute_ms = 0;
    const ExpertServeCfg& cf = *C.cfg;
    const uint64_t gpu0 = cf.gpu_rows != nullptr ? cf.gpu_rows(cf.gpu_user) : 0;
    for (;;) {
        uint32_t type = 0;
        if (!recv_frame(s, type, payload, kMaxReqFrame)) break;
        if (type == kBye) break;
        if (type != kReq) { send_err(s, "unexpected frame"); break; }
        np::ReqView v;
        const std::string why = np::parse_req(C.lim, payload.data(), payload.size(), v);
        if (!why.empty()) {   // refuse it, run nothing, keep the connection and the dispatch state as they were
            ++bad_reqs;
            if (bad_reqs <= 10 || bad_reqs % 1000 == 0) std::fprintf(stderr, "strata net: refused a request from %s: %s\n", peer.c_str(), why.c_str());
            if (!send_frame(s, kErr, why.data(), why.size())) break;
            continue;
        }
        out.assign(v.n * (size_t) H, 0.0f);
        const double t0 = now_ms();
        std::string why_failed;   // a failure answers one request; it must not poison the next (the compute callback resets its own state)
        if (!compute((int) v.layer, (int) v.n_tok, (int) v.k, (int) v.arith, v.ids, v.mask, v.x, out.data(), why_failed)) {
            const std::string m = std::string("expert pool failed: ") + (why_failed.empty() ? "?" : why_failed) + " (layer " + std::to_string(v.layer) + ")";
            if (!send_frame(s, kErr, m.data(), m.size())) break;
            continue;
        }
        const double us = (now_ms() - t0) * 1000.0;
        resp.clear();
        put<uint64_t>(resp, v.seq); put<double>(resp, us); put<uint32_t>(resp, v.rows); put<uint32_t>(resp, 0);
        const size_t at = resp.size();
        resp.resize(at + (size_t) v.rows * H * 4);
        size_t q = 0;
        for (size_t i = 0; i < v.n; ++i)
            if (v.mask[i]) { std::memcpy(resp.data() + at + q * (size_t) H * 4, out.data() + i * (size_t) H, (size_t) H * 4); ++q; }
        if (!send_frame(s, kResp, resp.data(), resp.size())) break;
        ++calls; compute_ms += us / 1000.0; rows_asked += v.rows;
    }
    sock_close(s);
    std::fprintf(stderr, "strata net: conductor %s disconnected after %llu calls (%llu refused), mean compute %.3f ms\n", peer.c_str(),
                 (unsigned long long) calls, (unsigned long long) bad_reqs, calls ? compute_ms / (double) calls : 0.0);
    if (cf.gpu_rows != nullptr) {
        const uint64_t on_gpu = cf.gpu_rows(cf.gpu_user) - gpu0;
        std::fprintf(stderr, "strata net: %llu rows asked, %llu computed on helper GPUs (%.1f%%)\n", (unsigned long long) rows_asked,
                     (unsigned long long) on_gpu, rows_asked ? 100.0 * (double) on_gpu / (double) rows_asked : 0.0);
    }
}

bool is_loopback(const sockaddr* a) {
    if (a->sa_family == AF_INET) return (ntohl(((const sockaddr_in*) a)->sin_addr.s_addr) >> 24) == 127;
    if (a->sa_family == AF_INET6) return IN6_IS_ADDR_LOOPBACK(&((const sockaddr_in6*) a)->sin6_addr) != 0;
    return false;
}
}  // namespace

int serve_experts_fn(const ExpertServeCfg& cfg, const ServeCompute& compute) {
    if (const std::string why = np::serve_cfg_problem(cfg.top_k, cfg.identity.fingerprint); !why.empty()) {
        std::fprintf(stderr, "strata net: the expert server is misconfigured: %s\n", why.c_str());
        return 1;
    }
    ignore_sigpipe();
    if (!sock_init()) return 1;
    const int lo = cfg.layer_lo, hi = cfg.layer_hi < 0 ? (int) cfg.n_layers - 1 : cfg.layer_hi;
    if (cfg.warm != nullptr) cfg.warm(cfg.warm_user, lo, hi);

    ServerCtx C;
    C.cfg = &cfg;
    C.token = env_token();
    C.tmo_ms = timeout_ms();
    C.H = cfg.n_embd > 0 ? cfg.n_embd : kFlashEmbd;
    C.lim.layer_lo = lo; C.lim.layer_hi = hi; C.lim.n_embd = C.H; C.lim.n_expert = cfg.n_expert; C.lim.top_k = cfg.top_k;
    C.lim.max_tok = strata::kernels::kWindowMaxT; C.lim.max_entries = strata::kernels::kWindowMaxEntries;
    C.lim.expert_lo = cfg.expert_lo; C.lim.expert_hi = cfg.expert_hi;   // B5: a request for an expert another host owns is refused
    ExpertServeCfg cfg2 = cfg;
    cfg2.layer_hi = hi;
    C.cfg = &cfg2;

    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string ps = std::to_string(cfg.port);
    if (getaddrinfo(cfg.host.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        std::fprintf(stderr, "strata net: cannot resolve the listen address %s\n", cfg.host.c_str());
        return 1;
    }
    sock_t ls = kBad;
    int bind_errno = 0;
    bool loopback = false;
    char bound[NI_MAXHOST] = "?";
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        ls = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (ls == kBad) continue;
#ifndef _WIN32
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);   // (Winsock's SO_REUSEADDR lets another process share the port)
#endif
        if (::bind(ls, a->ai_addr, (int) a->ai_addrlen) == 0 && ::listen(ls, 8) == 0) {
            loopback = is_loopback(a->ai_addr);
            getnameinfo(a->ai_addr, (socklen_t) a->ai_addrlen, bound, sizeof bound, nullptr, 0, NI_NUMERICHOST);
            break;
        }
        bind_errno = sock_errno();
        sock_close(ls);
        ls = kBad;
    }
    freeaddrinfo(res);
    if (ls == kBad) {
        std::fprintf(stderr, "strata net: cannot listen on %s:%d (errno %d)\n", cfg.host.c_str(), cfg.port, bind_errno);
        return 1;
    }
    std::fprintf(stderr, "strata net: expert server listening on %s:%d, layers %d-%d, experts %d-%d of %lld, auth %s, timeout %d ms\n", bound, cfg.port,
                 lo, hi, cfg.expert_lo, cfg.expert_hi < 0 ? (int) cfg.n_expert - 1 : cfg.expert_hi, (long long) cfg.n_expert,
                 C.token.empty() ? "off" : "token", C.tmo_ms);
    if (!loopback)
        std::fprintf(stderr, "strata net: WARNING: %s is not a loopback address: any host that can reach port %d may connect%s. Trusted LAN only: "
                             "set STRATA_EXPERT_TOKEN and firewall the port to the conductor's address\n", bound, cfg.port,
                     C.token.empty() ? " (NO TOKEN SET)" : "");

    std::thread acceptor([&] {
        std::list<std::unique_ptr<Conn>> conns;
        int backoff_ms = 0;
        auto reap = [&](bool all) {
            for (auto it = conns.begin(); it != conns.end();) {
                if (all || (*it)->done.load()) { if ((*it)->t.joinable()) (*it)->t.join(); it = conns.erase(it); }
                else ++it;
            }
        };
        while (!C.stop.load()) {
            reap(false);
            if (sock_poll(ls, false, 250) <= 0) continue;
            sock_t s = ::accept(ls, nullptr, nullptr);
            if (s == kBad) {   // EMFILE / ENFILE / ENOBUFS persist: back off instead of spinning
                backoff_ms = backoff_ms == 0 ? 50 : std::min(backoff_ms * 2, 1000);
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                continue;
            }
            backoff_ms = 0;
            if (C.handshaking.load() >= kMaxHandshakes) { sock_close(s); continue; }
            ++C.handshaking;
            auto c = std::make_unique<Conn>();
            Conn* cp = c.get();
            try {
                c->t = std::thread([&C, cp, s] { handshake_thread(&C, s); cp->done = true; });
                conns.push_back(std::move(c));
            } catch (const std::exception&) {
                sock_close(s);
                --C.handshaking;
            }
        }
        reap(true);
    });

    std::vector<uint8_t> payload;
    std::vector<float> out;
    std::vector<uint8_t> resp;
    while (!C.stop.load()) {
        Ready r;
        {
            std::unique_lock<std::mutex> lk(C.mu);
            C.cv.wait_for(lk, std::chrono::milliseconds(250), [&] { return C.ready.s != kBad; });
            if (C.ready.s == kBad) continue;
            r = C.ready;
            C.ready = Ready();
        }
        serve_session(compute, C, r.s, r.peer, payload, out, resp);
        C.busy = false;
        if (cfg.once) C.stop = true;
    }
    acceptor.join();
    {
        std::lock_guard<std::mutex> lk(C.mu);
        sock_close(C.ready.s);   // a conductor that finished its handshake as the server stopped
    }
    sock_close(ls);
    return 0;
}

}  // namespace strata::core
