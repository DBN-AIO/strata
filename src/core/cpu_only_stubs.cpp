// src/core/cpu_only_stubs.cpp - what the CUDA engine's .cu / driver-API sources provide, for the CPU-only expert server.
//
// strata-expert-server (src/program/expert_server_main.cpp) links expert_source.cpp, which is shared with the CUDA engine
// and so names a few symbols that live in GPU sources (pinned.cu, expert_cache.cpp's driver-API cache, the CUDA kernels).
// The server never reaches them: it runs `expert_pool_dispatch_multi` with no plan, no residency table, no GPU cache and no
// peer or helper GPU, so every row goes to the CPU pool.  The definitions below keep the link honest:
//   * fnv1a64 is the same function as pinned.cu's (a pure host hash), because it is cheap and must give the same bits;
//   * the cache and arena members answer "nothing resident" / do nothing;
//   * each GPU kernel entry point stops the process with a message naming itself, so a code path that should be
//     unreachable here fails loudly instead of returning zeros.
// A new GPU symbol in the shared sources fails the CPU-only link (CI: .github/workflows/build-linux.yml); add it here only
// after checking that the server cannot reach it.
#include "strata/core/expert_cache.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
[[noreturn]] void gpu_only(const char* what) {
    std::fprintf(stderr, "strata-expert-server: internal error: %s is a GPU path and this build has no GPU\n", what);
    std::abort();
}
}  // namespace

namespace strata::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// The pinned arena and its loaders are the ArenaExpertSource's (the expert server reads the GGUF in place instead).
PinnedArena::~PinnedArena() {}
PinnedArena::PinnedArena(uint64_t, const std::vector<uint64_t>&, uint64_t, const std::string&, uint64_t) { gpu_only("PinnedArena"); }
PinnedArena::PinnedArena(uint64_t, ByNeed, const std::vector<std::pair<uint64_t, uint64_t>>&, uint64_t, const std::string&,
                         uint64_t) {
    gpu_only("PinnedArena");
}

// ArenaExpertSource::open names these on the Windows path (#285 deferred registration; STRATA_ARENA_PIN_GIB): the CPU-only server
// maps the GGUF in place and never opens an arena, so they are never reached.
int arena_pin_cap_gib() { return -1; }
PinnedArena::PinnedArena(uint64_t, const std::vector<uint64_t>&, Deferred) { gpu_only("PinnedArena"); }
void PinnedArena::register_slices(std::atomic<int>&) { gpu_only("PinnedArena"); }

LoadStats load_experts_ranges(const std::string&, uint8_t*, const std::vector<uint64_t>&, const std::vector<uint64_t>&, int,
                              uint64_t, const std::atomic<int>*, const PadMap*) {
    LoadStats s;
    s.ok = false;
    s.error = "the arena loader is not in the CPU-only build";
    return s;
}

LoadStats load_experts_direct(const std::string&, uint8_t*, const std::vector<uint64_t>&, const std::vector<uint64_t>&, int,
                              uint64_t, const std::atomic<int>*) {
    LoadStats s;
    s.ok = false;
    return s;
}

bool experts_unbuffered(const std::vector<std::string>&, uint64_t, std::string& why, bool, uint64_t) {
    why = "not used by the CPU-only build";
    return false;
}

// The GPU expert cache: nothing is ever resident, nothing is admitted.
int32_t ExpertCache::slot_of(int64_t, int64_t) const { return -1; }
int32_t ExpertCache::admit(int64_t, int64_t) { return -1; }
bool ExpertCache::fill_slot(int32_t, const uint8_t*, void*, std::string& err, int64_t) {
    err = "no GPU expert cache in the CPU-only build";
    return false;
}

// Helper and peer GPUs: the dispatch calls them only when `d.remote_count` / `d.peer` are set, which the server never does.
bool RemoteExperts::begin(int64_t, const float*, const int32_t*, int64_t, int64_t, const int32_t*, const int32_t*, std::string& err) {
    err = "no helper GPU in the CPU-only build";
    return false;
}
bool RemoteExperts::finish(float*, std::string& err) {
    err = "no helper GPU in the CPU-only build";
    return false;
}
bool PeerExperts::launch(int64_t, const float*, const int32_t*, int64_t, int64_t, const int32_t*, std::string& err, float*) {
    err = "no peer GPU in the CPU-only build";
    return false;
}
bool PeerExperts::finish(float*, std::string& err) {
    err = "no peer GPU in the CPU-only build";
    return false;
}

}  // namespace strata::core

namespace strata::kernels {

void add_inplace(float*, const float*, int64_t, void*) { gpu_only("add_inplace"); }

void quantize_q8_0_scaled(const float*, uint8_t*, float*, int64_t, void*) { gpu_only("quantize_q8_0_scaled"); }

void moe_hit_grouped_s2(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*, void*, float*, void*,
                        const float*) {
    gpu_only("moe_hit_grouped_s2");
}

void moe_hit_grouped_s2_cpu_order(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*, void*, float*,
                                  void*, const float*, float*) {
    gpu_only("moe_hit_grouped_s2_cpu_order");
}

}  // namespace strata::kernels
