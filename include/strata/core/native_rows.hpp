// include/strata/core/native_rows.hpp - routed-expert rows on the CPU for the Qwen3.6 / Qwen3.8-Max family (qwen35moe), and the
// identity both ends of a remote-expert split compute from the model file.
//
// Flash-Next's expert server computes with the engine's ExpertDispatch over its pack layout (net_experts_dispatch.cpp).  The
// qwen35moe engine (src/qwen35/q36_engine.cu) keeps each expert as one blob [gate rows | up rows | down rows] in the GGUF's own
// block formats and runs ggml-cpu's dot products on them; `NativeRows` is that CPU half on its own (no CUDA): the same
// kernels, the same activation quantization, the same job grouping as the engine's hybrid mode, so a row computed by an expert
// server is bit-for-bit the row the conductor's own CPU pool would have produced (docs/QWEN_MAX.md).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/net_experts.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// What the two ends of a split must agree on, from the GGUF directory alone: per block (the trunk and the in-file MTP block)
/// the three expert tensors' types and shapes, the geometry (n_embd, experts, top-k) and the model files' sizes; plus the CPU
/// kernel path.  The same model file on both hosts gives the same bytes; another quantization or file is refused in the handshake.
NetIdentity native_net_identity(const strata::GgufModel& gm, const ModelGeometry& g, int64_t top_k, const std::vector<std::string>& files);

/// One block's experts in memory: expert e's blob at base + e * stride.
struct NativeLayerBlobs {
    const uint8_t* base = nullptr;   ///< null: this server does not hold the block
    size_t stride = 0;
    kernels::cpu::NativeFmt fmt;
};

/// Repack expert `e` of block `layer` of the GGUF into `dst` (fmt.bytes): gate rows, up rows, down rows, the engine's native blob.
/// False with `err` when the block's tensors are missing or their types do not give the layout `fmt` describes.
bool native_expert_blob(const strata::GgufModel& gm, int64_t layer, int64_t e, const kernels::cpu::NativeFmt& fmt, uint8_t* dst,
                        std::string& err);

class NativeRows {
public:
    NativeRows(std::vector<NativeLayerBlobs> layers, int64_t n_embd, int64_t n_expert, kernels::cpu::ExpertPool& pool)
        : layers_(std::move(layers)), n_embd_(n_embd), n_expert_(n_expert), pool_(pool) {}
    /// The rows of the entries i (token i / k, expert ids[i]) with mask[i] != 0 into rows + i * n_embd: x holds n_tok rows of
    /// n_embd floats.  Un-weighted down-projection outputs, as the engine's CPU pool writes them.
    bool run(int layer, int n_tok, int k, const int32_t* ids, const uint8_t* mask, const float* x, float* rows,
             kernels::cpu::NativeArithmetic arith, std::string& err);

private:
    std::vector<NativeLayerBlobs> layers_;
    int64_t n_embd_, n_expert_;
    kernels::cpu::ExpertPool& pool_;
    std::vector<uint8_t> acts_;
};

}  // namespace strata::core
