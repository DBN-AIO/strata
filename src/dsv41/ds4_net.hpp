// Strata. DeepSeek-V4.1-Flash experts over the LAN (docs/REMOTE_EXPERTS.md "DeepSeek-V4.1-Flash", docs/DEEPSEEK41_SPLIT.md).
//
// The cut point is the engine's CPU lane for one layer: `cpu_experts(layer, x, ents)`.  In: the layer's FFN input rows
// (n_tok x dim fp32) and the (token, expert) pairs the GPU does not hold.  Out: one un-weighted expert output row (dim fp32)
// per pair.  The weighted sum over the picks, the shared expert and the residual stay on the conductor, so the wire
// carries exactly what the qwen38 engine's does (src/core/net_experts.cpp, protocol SXP1 v2, unchanged except that the
// HELLO carries the engine family).
//
// This file is host only: no CUDA, no ggml headers.  The same code serves two programs:
//   strata-dsv41 --expert-server ...     (the CUDA engine binary; the server path returns before any CUDA call)
//   strata-dsv41-expert-server ...       (no CUDA at all: T340, and the stream PC until its toolkit exists)
// and the conductor side (Remote) is compiled into strata-dsv41.
#pragma once

#include "strata/core/net_experts.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4::net {

constexpr int kMaxTok = 16;                   ///< tokens a window holds: kWin in ds4_kernels.cuh (static_assert in ds4_model.cu)
constexpr int kMaxPicks = 8;                  ///< kPicks
constexpr int kMaxEntries = kMaxTok * kMaxPicks;
/// BBXL S5: tokens one PROMPT window sent to a server holds (the decode window stays kMaxTok).  The server's limits are these: n_tok up to
/// kMaxPromptTok, n_tok x k up to kMaxPromptEntries.  32 tokens at hidden 5120 / top-6 are a 655 KB request and a 3.9 MB answer, inside the wire's
/// frame limits (1 MiB / 4 MiB); `Remote` lowers its window for a larger hidden size.  An older server (limit 16) refuses a 32-token window: both ends of a prompt path need this build.
constexpr int kMaxPromptTok = 32;
constexpr int kMaxPromptEntries = kMaxPromptTok * kMaxPicks;
constexpr int kEntTok = 8;                    ///< tokens one expert takes in one kernel call (gg::kMxCpuMaxN)

/// ggml type ids of the expert tensors this path computes (the CPU lane's: MXFP4 and the ggml block types).
int type_of_name(const std::string& dtype);          ///< "IQ2_XXS" -> 16 ...; 0 when unknown
size_t type_row_bytes(int type, int64_t k);          ///< bytes of a row of k values; 0 when unknown or k is not a block multiple

/// One layer's routed experts as the pack lays them: matrix r (0 gate = w1, 1 up = w3, 2 down = w2) of expert e is at
/// `off[r] + e * stride[r]` in file `file[r]`, `bytes[r]` long.
struct ExpertLayer {
    int il = -1;                                      ///< -1: the pack has no experts for this layer
    int type[3] = {0, 0, 0};
    int file[3] = {-1, -1, -1};
    uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0}, bytes[3] = {0, 0, 0};
};

/// The pack's routed experts, read from its two text files (ds4.txt, tensors.txt) only: no engram, no dense weights, no
/// device.  Both ends build it, so both derive the same geometry and the same identity.
class ExpertMap {
public:
    std::string pack_dir, model_dir;
    int dim = 0, ff = 0, n_layers = 0, n_exp = 0, top_k = 0;
    float lim = 0.0f;
    std::vector<std::string> files;                   ///< the GGUF shards, as the pack names them
    std::vector<uint64_t> file_sizes;                 ///< their sizes on this host (0: not there)
    std::vector<ExpertLayer> layers;                  ///< index = layer
    bool load(const std::string& pack_dir, std::string& err);
    /// SHA-256 over the geometry, every layer's expert types / offsets / strides and the shards' sizes; arch dsv41; isa 0
    /// (the dsv41 CPU kernels give the same bits on every ISA).
    core::NetIdentity identity() const;
    std::string full_path(int file) const { return model_dir + "/" + files[(size_t) file]; }
};

/// The CPU expert kernels over the mapped shards; thread pool inside.  `compute` is what the server answers a REQ with and
/// what the conductor falls back to.
class ExpertHost {
public:
    ExpertHost();
    ~ExpertHost();
    ExpertHost(const ExpertHost&) = delete;
    ExpertHost& operator=(const ExpertHost&) = delete;
    /// Maps the shards of layers lo..hi (inclusive) read-only.  threads: 0 = one per hardware thread.
    bool open(const ExpertMap& m, int lo, int hi, int threads, std::string& err);
    /// Touches every page of the layers lo..hi so the page cache holds them; returns the bytes.
    uint64_t warm(int lo, int hi, int elo = 0, int ehi = -1);   ///< (elo..ehi: only those experts of each layer; -1 = to the last)
    /// Entries i of n_tok*k with mask[i] != 0: expert ids[i] on the token i / k (x + (i / k) * dim); the row goes to
    /// out + i * dim.  Same arithmetic as the engine's cpu_experts_iq / cpu_experts_mx.
    bool compute(uint32_t layer, uint32_t n_tok, uint32_t k, const int32_t* ids, const uint8_t* mask, const float* x, float* out,
                 std::string& err);
    uint64_t calls() const;
    double compute_ms() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// The conductor's side: sends a layer's CPU-lane entries to the servers and says which came back.
class Remote {
public:
    struct Item { int e = 0; int n = 0; int slot0 = 0; int tok[kMaxTok] = {}; };   ///< the engine's Ent
    Remote();
    ~Remote();
    /// spec = "host:port=lo-hi[/e0-e1][,...]".  A server that cannot be reached or does not match is an error here.
    bool open(const ExpertMap& m, const std::string& spec, std::string& err, uint32_t arch_override = 0xffffffffu);
    bool active() const;
    bool serves(int layer) const;
    /// BBXL S7: a server owns expert `e` (the router's original id; the S3 renumbering is applied here) of `layer`.
    bool owns(int layer, int e) const;
    /// Runs the items on the servers.  remote[q] = 1: item q's rows are in y[(slot0 + i) * dim] (i < n).  Items left at 0
    /// (not owned by a route, the route down, an error) are the caller's to compute.  Always true unless the request itself
    /// cannot be built (then nothing is sent and every item stays local).
    bool run(int layer, const float* x, size_t ldx, const Item* items, int n_items, float* y, std::vector<char>& remote);
    /// BBXL S4: run() in two halves, so the conductor's own experts compute while the servers answer.  begin() writes every route's
    /// request for the layer (all routes at once, before any answer is read) and returns: pending[q] = 1 means item q's rows are on
    /// their way; items at 0 are the caller's (no route owns them) - compute them NOW, then call finish().  finish() reads every
    /// answer and writes the rows of the pending items to y[(slot0 + i) * dim]; remote[q] = 1 means they are there, a pending item
    /// left at 0 lost its host and is the caller's too.  Same rows, bit for bit, as run().  At most one layer in flight.
    bool begin(int layer, const float* x, size_t ldx, const Item* items, int n_items, std::vector<char>& pending);
    bool finish(float* y, std::vector<char>& remote);
    bool in_flight() const;
    /// BBXL S5, the prompt path: T tokens of one layer (any T; the caller slices a long prompt into slabs) through the servers that own the
    /// layer's experts, in windows of up to `prompt_window()` tokens with several in flight on every host (`NetExperts::prompt`).
    /// x: the T rows of the layer's FFN input, ldx apart; ids: T x top_k expert ids as the router picked them (the S3 renumbering is applied
    /// here).  rows: T x top_k x dim fp32, the un-weighted expert rows; got[i] = 1 where row i is there.  Entries no route owns (the
    /// conductor's own experts) and every entry after a failure (the route is dropped and retried later, as in decode) stay at 0: the caller
    /// computes those from its own copy.  Returns false only when a request failed (got is then all 0); `err` says why.
    bool prompt(int layer, const float* x, size_t ldx, int T, const int32_t* ids, float* rows, std::vector<uint8_t>& got, std::string& err);
    int prompt_window() const;   ///< tokens a prompt window holds (STRATA_DS4_PROMPT_WIN lowers it, for measuring)
    uint64_t rows_remote() const;
    uint64_t rows_local() const;
    uint64_t calls() const;   ///< requests sent to the servers: one per run() that reached a route (a window of any n tokens is one)
    std::string stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// `--expert-server` for either binary: parses --pack --listen [HOST:]PORT --serve-layers A-B --cpu-threads N --no-warm and
/// the engine flags a server ignores, then serves until stopped.  argv[0] is skipped.
int server_main(int argc, char** argv);

}  // namespace strata::ds4::net
