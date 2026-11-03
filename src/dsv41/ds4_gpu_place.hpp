// Strata. Where a GPU expert server puts the experts it serves (docs/DEEPSEEK41_SPLIT.md, "GPU expert server").
// Host only, no CUDA: the fit and placement rules are plain functions, so they are tested without a card
// (tests/dsv41/gpu_place_test.cpp) and the CUDA side (ds4_net_gpu.cu) only carries the plan out.
#pragma once

#include "ds4_net.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4::net {

constexpr size_t kGpuAlign = 256;   ///< every matrix of an expert's VRAM blob, and the blob itself, starts on a multiple of this

/// One expert's blob in VRAM: gate, up, down one after another (each at off[r], mat[r] bytes of it), stride bytes in all.
struct GpuBlob {
    size_t off[3] = {0, 0, 0}, mat[3] = {0, 0, 0}, stride = 0;
};
inline bool gpu_blob_layout(const ExpertMap&, int, GpuBlob&, std::string& why) { why = "not implemented"; return false; }

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
inline GpuPlace gpu_plan(const GpuPlaceIn&) { GpuPlace p; p.why = "not implemented"; return p; }

}  // namespace strata::ds4::net
