// include/strata/core/net_experts.hpp - remote experts over the LAN (expert parallelism across PCs).
//
// The conductor keeps attention, the router, the shared expert and the GPU's expert cache.  For the layers it was
// told to hand out (--remote-experts host:port=lo-hi) it ships the layer's MoE input (n_tok x n_embd fp32) and the
// routing of the entries its own GPU does not hold to an expert server; the server computes those experts with the
// same CPU kernels the conductor would have used and returns the un-weighted expert output rows.  The conductor
// writes them into the same rows the local CPU pool would have filled, so everything after the pool is unchanged.
// If the server fails (down, hung, wrong answer) the conductor computes those rows itself and reconnects later.
//
// Wire protocol "SXP1", version 2 (all integers little endian, one persistent TCP connection, TCP_NODELAY):
//   frame   = u32 magic 0x31505853 | u32 type | u64 payload_len | payload          (16 byte header)
//   CHALLENGE(7) server->client  u32 version(2) | u32 flags(bit0: a token is required) | u8 nonce[32]
//   HELLO(1)  client->server  u32 version(2) | u32 n_embd | u32 n_expert | u32 n_layers | u32 isa | u32 arch |
//                             u8 model_fingerprint[32] | u8 mac[32]
//                             mac = HMAC-SHA256(token, nonce || the 56 HELLO bytes before the mac), zeros without a token
//                             arch (was a zero pad): the engine family, kArchQwen38 = 0 (the old pad value, so an older conductor still
//                             matches a qwen38 server) or kArchDsv41 = 1.  A server refuses a conductor of another arch.
//   HELLO_ACK(2) server->client u32 version(2) | u32 ok(1) | u32 layer_lo | u32 layer_hi | u32 n_embd | u32 n_expert |
//                             u32 top_k | u32 isa
//   REQ(3)    client->server  u32 layer | u32 n_tok | u32 k | u32 arith | u64 seq |
//                             i32 ids[n_tok*k] | u8 mask[n_tok*k] (padded to 4) | f32 x[n_tok*n_embd]
//   RESP(4)   server->client  u64 seq | f64 server_us | u32 n_rows | u32 pad | f32 rows[n_rows*n_embd]
//                             (rows in ascending entry index of the mask)
//   ERR(5)    server->client  utf-8 text.  After a refused HELLO the server closes; after a refused REQ it keeps serving.
//   BYE(6)    either
// Trust model: a TRUSTED LAN.  The token (env STRATA_EXPERT_TOKEN, both sides) authenticates the conductor to the
// server (challenge-response, the token never crosses the wire); the traffic itself is neither encrypted nor
// authenticated afterwards.  docs/REMOTE_EXPERTS.md "Security".
// Sockets sit behind one small shim (POSIX or Winsock) in net_experts.cpp; nothing else in the path is OS specific.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {
struct ExpertDispatch;

/// What the two ends must agree on for "bit identical": the model's expert tensors and the CPU kernel path.
inline constexpr uint32_t kArchQwen38 = 0;   ///< src/core + src/program: Qwen3.8-Flash-Next and the qwen35moe family
inline constexpr uint32_t kArchDsv41 = 1;    ///< src/dsv41: DeepSeek-V4.1-Flash (docs/DEEPSEEK41_SPLIT.md)
const char* net_arch_name(uint32_t arch);
struct NetIdentity {
    uint8_t fingerprint[32] = {};   ///< SHA-256 over the expert layout (types, offsets, sizes) and the model files' sizes
    uint32_t isa = 0;               ///< the CPU kernel path (net_isa_name); the dsv41 kernels are bit-exact on every ISA: 0
    uint32_t arch = kArchQwen38;    ///< the engine family; carried in HELLO, a mismatch is refused by the server
    uint32_t n_embd = 0;            ///< the hidden size on the wire; 0 = the qwen38 engine's compile-time H
};
/// `model_files`: the GGUF / experts.bin files the experts are read from (their sizes enter the fingerprint).
NetIdentity net_identity(const std::vector<std::string>& model_files);
std::string net_isa_name(uint32_t isa);
/// The CPU kernel path word of NetIdentity::isa (avx512 / avx512bw / avx2 bits and the STRATA_FORCE_ISA cap).  Not in the dsv41 build (STRATA_NET_NO_QWEN).
uint32_t net_isa_bits();
/// Socket read/write/connect timeout in ms for both ends (default 10000; env STRATA_EXPERT_TIMEOUT_MS; --expert-timeout-ms).
void net_set_timeout_ms(int ms);

/// Not thread safe: one conductor thread calls begin / finish (one request in flight, `cur_`, `owned_`, `buf_`).
class NetExperts {
public:
    NetExperts() = default;
    ~NetExperts();
    NetExperts(const NetExperts&) = delete;
    NetExperts& operator=(const NetExperts&) = delete;

    /// spec = "host:port=lo-hi[/e0-e1][,...]" (port defaults to 7840, lo-hi inclusive layers, host may be [v6]; e0-e1: the
    /// experts this host owns for those layers, default all).  Hosts may share layers with disjoint expert ranges: a layer's
    /// request then goes to all of its hosts before any answer is read (B5).
    /// Connects to every server once; a failure here is an error (the user asked for it).  After that a server that fails is
    /// dropped and retried with backoff while its rows are computed locally.
    bool open(const std::string& spec, int64_t n_layers, int64_t n_expert, const NetIdentity& id, std::string& err);
    bool active() const { return !routes_.empty(); }
    bool serves(int64_t layer) const { return route_of(layer) >= 0; }
    /// The hosts of `layer` together own every expert (B5: the prompt path hands a layer to the servers only then).
    bool covers(int64_t layer) const;
    /// Ship the entries whose kind is -1 (the CPU's) to the server for `layer`; those entries get kind 3.  When the route
    /// is down (or the send fails) nothing is sent, no kind changes, and the caller's CPU work covers the rows: always true.
    /// A request was sent exactly when in_flight().
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
               int arith, std::string& err);
    bool in_flight() const { return cur_ >= 0; }
    /// Receive the rows of the request begun last and write them into `out` (n_tok*k rows of n_embd).  On false the route is
    /// dropped and `err` says why; the caller must compute owned() locally.
    bool finish(float* out, std::string& err);
    /// The entries (indices into n_tok*k) of the request begun last; after a false finish(), those of the hosts that failed.
    const std::vector<int32_t>& owned() const { return owned_; }
    /// The caller leaves between begin and finish (an error elsewhere): the unread RESP would desync the stream, so drop the route.
    void abandon();
    /// The prompt path (prefill, B3): compute the entries i with mask[i] != 0 of T tokens (ids: T*k routed ids, x: T rows of
    /// n_embd) on the server of `layer`, in windows of at most kWindowMaxT tokens (the server's limit), and copy entry i's row
    /// to dst[i].  Synchronous, on the prompt thread (never while a decode request is in flight).  False with `err` when the
    /// route is down or fails (it is then dropped and retried later, as in decode): the caller computes those rows itself.
    bool prompt(int64_t layer, const float* x, const int32_t* ids, int64_t T, int64_t k, const uint8_t* mask,
                float* const* dst, std::string& err);

private:
    struct Route { std::string host; int port = 7840; int lo = 0, hi = 0; int elo = 0, ehi = 0; intptr_t fd = -1; int top_k = 0;   // top_k: the server's, from its HELLO_ACK; elo..ehi: the experts it owns
                   uint64_t calls = 0, bytes_out = 0, bytes_in = 0; double rtt_ms = 0, server_ms = 0, rtt_max_ms = 0;
                   // fault handling: down since an error; retry at retry_at_ms (steady clock), backoff doubles to 30 s
                   bool down_logged = false; double retry_at_ms = 0; int backoff_ms = 0;
                   uint64_t outages = 0, local_rows = 0, reconnects = 0;
                   // the prompt path (prompt()): windows, rows, wall time of the calls and the server's compute
                   uint64_t p_calls = 0, p_rows = 0, p_bytes_out = 0, p_bytes_in = 0; double p_ms = 0, p_server_ms = 0; };
    int route_of(int64_t layer) const {
        for (size_t i = 0; i < routes_.size(); ++i) if (layer >= routes_[i].lo && layer <= routes_[i].hi) return (int) i;
        return -1;
    }
    bool connect_route(Route& r, int timeout_ms, std::string& err);
    void mark_down(Route& r, const std::string& why);
    bool route_up(Route& r);
    bool send_req(Route& r, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const uint8_t* mask,
                  int arith, uint64_t& seq, std::vector<uint8_t>& buf);
    std::string recv_resp(Route& r, uint64_t seq, uint32_t want, std::vector<uint8_t>& resp, double& server_us);
    /// B5: per route, the decode request in flight (several hosts may hold parts of one layer)
    struct InFlight { bool active = false; uint64_t seq = 0; double t_send = 0; std::vector<uint8_t> mask, buf; std::vector<int32_t> owned; };
    std::vector<InFlight> inflight_;
    std::vector<uint8_t> resp_;
    uint64_t layer_calls_ = 0, multi_calls_ = 0;
    double wait_ms_ = 0;
    std::vector<Route> routes_;
    int64_t n_layers_ = 0, n_expert_ = 0;
    NetIdentity id_;
    std::string token_;
    // the request in flight
    int cur_ = -1;
    uint64_t seq_ = 0;
    int64_t cur_n_ = 0;
    std::vector<uint8_t> mask_, buf_;
    std::vector<int32_t> owned_;
    double t_send_ms_ = 0;
    int64_t H_ = 0;   ///< hidden size on the wire (id_.n_embd, else the qwen38 engine's H)
};

struct ExpertServeCfg {
    std::string host = "127.0.0.1";    ///< bind address: loopback unless the user writes another one (0.0.0.0 = every interface)
    int port = 7840;
    int layer_lo = 0, layer_hi = -1;   ///< served layers (inclusive); hi < 0: all
    int expert_lo = 0, expert_hi = -1; ///< served experts of those layers (inclusive, --serve-experts); hi < 0: all
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    int64_t top_k = 0;                 ///< the model's routed experts per token: a request with another k is refused
    NetIdentity identity;
    bool once = false;                 ///< serve one connection, then return
    /// Called once at start with (lo, hi): make the served layers' experts resident in RAM (warm the source).
    void (*warm)(void* user, int lo, int hi) = nullptr;
    void* warm_user = nullptr;
    /// --expert-server-gpu: the dispatch may have helper GPUs (RemoteExperts) holding served experts; their rows are the GPU
    /// kernels' (not bit-identical to the conductor's CPU rows).  Off: a CPU-only dispatch, as before.
    bool gpu = false;
    /// Optional: rows computed so far by GPU expert tiers of this server (helper GPUs), for the per-connection summary.
    uint64_t (*gpu_rows)(void* user) = nullptr;
    void* gpu_user = nullptr;
    /// Window limits a request is checked against (rows, and rows x k); 0: the qwen38 engine's (window_limits.hpp).
    int64_t max_tok = 0, max_entries = 0;
};

/// What an expert server computes for one request: the entries i of n_tok*k whose mask[i] is 1 (expert ids[i] for the token
/// i / k, whose hidden state is x + (i / k) * n_embd), written to out + i * n_embd (un-weighted FFN output, fp32).  `arith`
/// is the request's arithmetic mode (0 default, 1 window invariant).  False with `err`: answered with ERR, the connection stays.
using ExpertComputeFn = std::function<bool(uint32_t layer, uint32_t n_tok, uint32_t k, int arith, const int32_t* ids,
                                           const uint8_t* mask, const float* x, float* out, std::string& err)>;
/// The expert server loop for any engine family: sockets, handshake (token, arch, geometry, fingerprint), one conductor at a time.
int serve_experts(const ExpertServeCfg& cfg, const ExpertComputeFn& compute);

/// The expert server loop: answers REQ frames with `d` (an ordinary pool dispatch with every row on the CPU).  `d` must
/// be a CPU-only dispatch (no peer GPU, no helper GPUs, no plan, no net): refused with 1 otherwise.  A request is validated
/// before it runs; a refused one is answered with ERR and the connection keeps serving.  Returns 0 on a clean stop.
int serve_experts(ExpertDispatch& d, const ExpertServeCfg& cfg);

}  // namespace strata::core
