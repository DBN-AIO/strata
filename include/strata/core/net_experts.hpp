// include/strata/core/net_experts.hpp - remote experts over the LAN (expert parallelism across PCs).
//
// The conductor keeps attention, the router, the shared expert and the GPU's expert cache.  For the layers it was
// told to hand out (--remote-experts host:port=lo-hi) it ships the layer's MoE input (n_tok x n_embd fp32) and the
// routing of the entries its own GPU does not hold to an expert server; the server computes those experts with the
// same CPU kernels the conductor would have used and returns the un-weighted expert output rows.  The conductor
// writes them into the same rows the local CPU pool would have filled, so everything after the pool is unchanged.
//
// Wire protocol "SXP1" (all integers little endian, one persistent TCP connection, TCP_NODELAY):
//   frame   = u32 magic 0x31505853 | u32 type | u64 payload_len | payload
//   HELLO(1)  client->server  u32 version(1) | u32 n_embd | u32 n_expert | u32 n_layers
//   HELLO_ACK(2) server->client u32 ok | u32 layer_lo | u32 layer_hi | u32 n_embd | u32 n_expert
//                             [| u32 expert_lo | u32 expert_hi]  (B5; a 20-byte ack means every expert)
//   REQ(3)    client->server  u32 layer | u32 n_tok | u32 k | u32 arith | u64 seq |
//                             i32 ids[n_tok*k] | u8 mask[n_tok*k] (padded to 4) | f32 x[n_tok*n_embd]
//   RESP(4)   server->client  u64 seq | f64 server_us | u32 n_rows | u32 pad | f32 rows[n_rows*n_embd]
//                             (rows in ascending entry index of the mask)
//   ERR(5)    server->client  utf-8 text
//   BYE(6)    either
// Sockets sit behind one small shim (POSIX or Winsock) in net_experts.cpp; nothing else in the path is OS specific.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
struct ExpertDispatch;

class NetExperts {
public:
    NetExperts() = default;
    ~NetExperts();
    NetExperts(const NetExperts&) = delete;
    NetExperts& operator=(const NetExperts&) = delete;

    /// spec = "host:port=lo-hi[/e0-e1][,host:port=lo-hi[/e0-e1]...]" (port defaults to 7840; lo-hi inclusive layers;
    /// e0-e1 the experts this host owns for them, default all).  Hosts may share layers when their expert ranges are
    /// disjoint: a layer's request then goes to all of them at once (B5, request pipelining across hosts).
    bool open(const std::string& spec, int64_t n_layers, int64_t n_expert, std::string& err);
    bool active() const { return !routes_.empty(); }
    /// Some host owns experts of `layer` (decode: their CPU entries go there, the rest stay local).
    bool serves(int64_t layer) const { return route_of(layer) >= 0; }
    /// The hosts of `layer` together own every expert (the prompt path sends a layer only then).
    bool covers(int64_t layer) const {
        std::vector<uint8_t> own((size_t) n_expert_, 0);
        for (const Route& r : routes_)
            if (layer >= r.lo && layer <= r.hi) for (int e = r.elo; e <= r.ehi; ++e) own[(size_t) e] = 1;
        for (uint8_t o : own) if (!o) return false;
        return n_expert_ > 0;
    }
    /// Ship the entries whose kind is -1 (the CPU's) to the hosts that own their experts for `layer`, all before any
    /// answer is read; those entries get kind 3.  Returns true with nothing sent when no host owns such an entry.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
               int arith, std::string& err);
    /// Receive the rows of the requests begun last and write them into `out` (n_tok*k rows of n_embd).
    bool finish(float* out, std::string& err);
    /// The prompt path (prefill): compute the entries i with mask[i] != 0 of T tokens (ids: T*k routed ids, x: T rows
    /// of n_embd) on the hosts for `layer`, in windows of at most kWindowMaxT tokens (the server's limit), up to
    /// STRATA_NET_PIPELINE windows in flight per host (default 4), and copy entry i's row to dst[i].
    bool prompt(int64_t layer, const float* x, const int32_t* ids, int64_t T, int64_t k, const uint8_t* mask,
                float* const* dst, std::string& err);

private:
    struct Route { std::string host; int port = 7840; int lo = 0, hi = 0; int elo = 0, ehi = 0; intptr_t fd = -1;
                   uint64_t calls = 0, bytes_out = 0, bytes_in = 0; double rtt_ms = 0, server_ms = 0, rtt_max_ms = 0;
                   uint64_t p_calls = 0, p_rows = 0, p_bytes_out = 0, p_bytes_in = 0; double p_ms = 0, p_server_ms = 0; };
    struct InFlight { bool active = false; uint64_t seq = 0; double t_send = 0;
                      std::vector<uint8_t> mask, buf; std::vector<int32_t> owned; };
    int route_of(int64_t layer) const {
        for (size_t i = 0; i < routes_.size(); ++i) if (layer >= routes_[i].lo && layer <= routes_[i].hi) return (int) i;
        return -1;
    }
    bool send_req(Route& r, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                  const uint8_t* mask, int arith, uint64_t& seq, std::vector<uint8_t>& buf, std::string& err);
    bool recv_resp(Route& r, uint64_t seq, uint32_t want, std::vector<uint8_t>& resp, double& server_us, std::string& err);
    std::vector<Route> routes_;
    std::vector<InFlight> inflight_;   ///< per route: the decode request in flight
    std::vector<uint8_t> resp_;
    int64_t n_expert_ = 0;
    uint64_t seq_ = 0;
    bool any_inflight_ = false;
    double t_send_ms_ = 0;
    uint64_t layer_calls_ = 0, multi_calls_ = 0;
    double wait_ms_ = 0;
};

struct ExpertServeCfg {
    int port = 7840;
    int layer_lo = 0, layer_hi = -1;   ///< served layers (inclusive); hi < 0: all
    int expert_lo = 0, expert_hi = -1; ///< served experts of those layers (inclusive, --serve-experts); hi < 0: all
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    bool once = false;                 ///< serve one connection, then return
    /// Called once at start with (lo, hi): make the served layers' experts resident in RAM (warm the source).
    void (*warm)(void* user, int lo, int hi) = nullptr;
    void* warm_user = nullptr;
    /// Optional: rows computed so far by GPU expert tiers of this server (helper GPUs), for the per-connection summary.
    uint64_t (*gpu_rows)(void* user) = nullptr;
    void* gpu_user = nullptr;
};

/// The expert server loop: answers REQ frames with `d` (an ordinary pool dispatch with every row on the CPU).
/// Returns 0 on a clean stop.
int serve_experts(ExpertDispatch& d, const ExpertServeCfg& cfg);

}  // namespace strata::core
