// Strata. The DeepSeek-V4.1-Flash GPU expert server's compute side: the served layers' experts held in VRAM, answered with the
// engine's own window kernels (ds4_gguf.cu win_rows, over ds4_iq.cu / the MXFP4 kernels) instead of the CPU lane
// (docs/DEEPSEEK41_SPLIT.md, "GPU expert server").  This header is host only (no CUDA types); the implementation is ds4_net_gpu.cu,
// compiled into strata-dsv41 (and the GPU tests) only.  The fit rules are ds4_gpu_place.hpp.
#pragma once

#include "ds4_gpu_place.hpp"
#include "ds4_net.hpp"

#include <memory>
#include <string>

namespace strata::ds4::net {

struct GpuOpts {
    int device = 0;                        ///< CUDA device ordinal (pin a card with CUDA_VISIBLE_DEVICES=<uuid>, device 0 then)
    uint64_t cap_bytes = 0;                ///< --gpu-vram-gb: most VRAM the experts may take; 0: free VRAM minus the reserve
    uint64_t reserve_bytes = 256ull << 20; ///< --gpu-reserve-mib: left free after the experts, the work buffers and the kernels' scratch
    bool cpu_rest = false;                 ///< --gpu-cpu-rest: layers that do not fit (or have a type with no device kernel) stay on the CPU lane
};

class GpuExpertHost {
public:
    GpuExpertHost();
    ~GpuExpertHost();
    GpuExpertHost(const GpuExpertHost&) = delete;
    GpuExpertHost& operator=(const GpuExpertHost&) = delete;
    /// Plans (ds4_gpu_place.hpp) and uploads experts elo..ehi (-1: the last) of layers lo..hi.  false with `err`: nothing fits
    /// (the message gives the shortfall), a layer has a type the device kernels do not take (without cpu_rest), or CUDA failed.
    bool open(const ExpertMap& m, int lo, int hi, int elo, int ehi, const GpuOpts& o, std::string& err);
    bool holds(int layer) const;           ///< the layer's served experts are in VRAM
    int layers_on_gpu() const;
    uint64_t resident_bytes() const;       ///< expert bytes in VRAM
    const GpuPlace& plan() const;
    std::string device_name() const;
    /// Same contract as ExpertHost::compute (ds4_net.hpp): the rows of the entries with mask[i] != 0 go to out + i * dim.
    bool compute(uint32_t layer, uint32_t n_tok, uint32_t k, const int32_t* ids, const uint8_t* mask, const float* x, float* out,
                 std::string& err);
    uint64_t calls() const;
    uint64_t rows() const;                 ///< rows computed on the card
    double compute_ms() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4::net
