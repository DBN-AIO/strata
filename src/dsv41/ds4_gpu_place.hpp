// Strata. Where a GPU expert server puts the experts it serves (docs/DEEPSEEK41_SPLIT.md, "GPU expert server").
// Host only, no CUDA: the fit and placement rules are plain functions, so they are tested without a card
// (tests/dsv41/gpu_place_test.cpp) and the CUDA side (ds4_net_gpu.cu) only carries the plan out.
#pragma once

#include "ds4_net.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::ds4::net {

constexpr size_t kGpuAlign = 256;   ///< every matrix of an expert's VRAM blob, and the blob itself, starts on a multiple of this

/// One expert's blob in VRAM: gate, up, down one after another (each at off[r], mat[r] bytes of it), stride bytes in all.
struct GpuBlob {
    size_t off[3] = {0, 0, 0}, mat[3] = {0, 0, 0}, stride = 0;
};
inline size_t gpu_round(size_t n) { return (n + kGpuAlign - 1) / kGpuAlign * kGpuAlign; }
inline bool gpu_blob_layout(const ExpertMap& m, int il, GpuBlob& b, std::string& why) {
    b = GpuBlob();
    if (il < 0 || il >= (int) m.layers.size() || m.layers[(size_t) il].il < 0) { why = "layer " + std::to_string(il) + " has no routed experts in the pack"; return false; }
    const ExpertLayer& L = m.layers[(size_t) il];
    if (L.type[0] != L.type[1]) { why = "layer " + std::to_string(il) + ": gate and up must have one type"; return false; }
    // gate / up: ff rows of dim values; down: dim rows of ff values
    const size_t g = type_row_bytes(L.type[0], m.dim), d = type_row_bytes(L.type[2], m.ff);
    if (g == 0 || d == 0) {
        why = "layer " + std::to_string(il) + ": expert type " + std::to_string(g == 0 ? L.type[0] : L.type[2]) + " has no known row layout for this geometry";
        return false;
    }
    b.mat[0] = g * (size_t) m.ff; b.mat[1] = b.mat[0]; b.mat[2] = d * (size_t) m.dim;
    size_t at = 0;
    for (int r = 0; r < 3; ++r) { b.off[r] = at; at += gpu_round(b.mat[r]); }
    b.stride = at;
    return true;
}

struct GpuPlaceIn {
    std::vector<uint64_t> layer_bytes;   ///< per served layer, in order: the bytes its served experts take in VRAM
    uint64_t vram_free = 0;              ///< free VRAM on the card now
    uint64_t cap = 0;                    ///< --gpu-vram-gb as bytes; 0: no cap
    uint64_t reserve = 0;                ///< kept free for the context, the scratch and the window buffers
    bool cpu_rest = false;               ///< --gpu-cpu-rest: a layer that does not fit is served by the CPU lane
};
struct GpuPlace {
    bool ok = false;
    std::vector<char> on_gpu;            ///< per layer: 1 = in VRAM
    uint64_t budget = 0, need = 0, placed = 0, shortfall = 0;
    std::string why;                     ///< when !ok (or a partial placement): what to tell the operator
};
inline std::string gpu_gib(uint64_t b) { char t[48]; std::snprintf(t, sizeof t, "%.2f GiB", (double) b / 1073741824.0); return t; }

/// The budget is the free VRAM minus the reserve, lowered to the cap when there is one.  Everything fits: all layers on the card.
/// Otherwise refused with the shortfall, or (cpu_rest) first-fit in layer order with the rest left to the CPU lane; a plan that
/// puts nothing on the card is refused either way (it would be a CPU server that says it is a GPU one).
inline GpuPlace gpu_plan(const GpuPlaceIn& in) {
    GpuPlace p;
    p.on_gpu.assign(in.layer_bytes.size(), 0);
    if (in.layer_bytes.empty()) { p.why = "no layers to serve"; return p; }
    for (uint64_t b : in.layer_bytes) p.need += b;
    p.budget = in.vram_free > in.reserve ? in.vram_free - in.reserve : 0;
    if (in.cap > 0 && in.cap < p.budget) p.budget = in.cap;
    const std::string have = "VRAM budget " + gpu_gib(p.budget) + " (free " + gpu_gib(in.vram_free) + ", reserve " + gpu_gib(in.reserve) +
                             (in.cap > 0 ? ", cap " + gpu_gib(in.cap) : std::string()) + ")";
    if (p.need <= p.budget) {
        std::fill(p.on_gpu.begin(), p.on_gpu.end(), (char) 1);
        p.placed = p.need;
        p.ok = true;
        return p;
    }
    p.shortfall = p.need - p.budget;
    if (!in.cpu_rest) {
        p.why = "the served experts need " + gpu_gib(p.need) + ", " + have + ": short by " + gpu_gib(p.shortfall) +
                "; serve fewer layers (--serve-layers / --serve-experts), free VRAM, or --gpu-cpu-rest to leave the rest to the CPU";
        return p;
    }
    uint64_t left = p.budget;
    for (size_t i = 0; i < in.layer_bytes.size(); ++i)
        if (in.layer_bytes[i] <= left) { p.on_gpu[i] = 1; left -= in.layer_bytes[i]; p.placed += in.layer_bytes[i]; }
    if (p.placed == 0) {
        p.why = "no served layer fits: " + have + " against the smallest layer's " +
                gpu_gib(*std::min_element(in.layer_bytes.begin(), in.layer_bytes.end())) + ": short by " + gpu_gib(p.shortfall);
        return p;
    }
    p.ok = true;
    p.why = "partial: " + gpu_gib(p.placed) + " of " + gpu_gib(p.need) + " in VRAM, the rest on the CPU lane (" + have + ")";
    return p;
}

}  // namespace strata::ds4::net
