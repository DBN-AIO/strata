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

    /// spec = "host:port=lo-hi[,host:port=lo-hi...]" (port defaults to 7840, lo-hi inclusive layers)
    bool open(const std::string& spec, int64_t n_layers, int64_t n_expert, std::string& err);
    bool active() const { return !routes_.empty(); }
    bool serves(int64_t layer) const { return route_of(layer) >= 0; }
    /// Ship the entries whose kind is -1 (the CPU's) to the server for `layer`; those entries get kind 3.
    /// Returns true with nothing sent when the layer has no such entry.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
               int arith, std::string& err);
    /// Receive the rows of the request begun last and write them into `out` (n_tok*k rows of n_embd).
    bool finish(float* out, std::string& err);

private:
    struct Route { std::string host; int port = 7840; int lo = 0, hi = 0; intptr_t fd = -1;
                   uint64_t calls = 0, bytes_out = 0, bytes_in = 0; double rtt_ms = 0, server_ms = 0, rtt_max_ms = 0; };
    int route_of(int64_t layer) const {
        for (size_t i = 0; i < routes_.size(); ++i) if (layer >= routes_[i].lo && layer <= routes_[i].hi) return (int) i;
        return -1;
    }
    std::vector<Route> routes_;
    int64_t n_expert_ = 0;
    // the request in flight
    int cur_ = -1;
    uint64_t seq_ = 0;
    int64_t cur_n_ = 0;
    std::vector<uint8_t> mask_, buf_;
    std::vector<int32_t> owned_;
    double t_send_ms_ = 0;
};

struct ExpertServeCfg {
    int port = 7840;
    int layer_lo = 0, layer_hi = -1;   ///< served layers (inclusive); hi < 0: all
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    bool once = false;                 ///< serve one connection, then return
    /// Called once at start with (lo, hi): make the served layers' experts resident in RAM (warm the source).
    void (*warm)(void* user, int lo, int hi) = nullptr;
    void* warm_user = nullptr;
};

/// The expert server loop: answers REQ frames with `d` (an ordinary pool dispatch with every row on the CPU).
/// Returns 0 on a clean stop.
int serve_experts(ExpertDispatch& d, const ExpertServeCfg& cfg);

}  // namespace strata::core
