// src/core/net_experts_dispatch.cpp - Flash-Next's expert server: the model-independent loop (net_experts.cpp,
// serve_experts with a callback) with the engine's CPU pool dispatch (expert_pool_dispatch_multi) as its compute.
#include "strata/core/expert_source.hpp"
#include "strata/core/net_experts.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdio>

namespace strata::core {

int serve_experts(ExpertDispatch& d, const ExpertServeCfg& cfg) {
    // The server answers with the plain CPU pool.  A second GPU, helper GPUs, a resident plan or a net route would compute
    // some rows on other arithmetic than the conductor's CPU kernels (and a net route would forward the request again).
    // --expert-server-gpu (cfg.gpu) lets helper GPUs compute the experts they hold; they take only kind -1 entries, i.e. the
    // asked-for ones (only_mask), and write into the same rows.
    if (d.peer != nullptr || (d.remote_count != 0 && !cfg.gpu) || d.net != nullptr || d.plan != nullptr || d.host_res != nullptr) {
        std::fprintf(stderr, "strata net: the expert server needs a CPU-only dispatch (no peer GPU, helper GPUs, resident plan or net route); "
                             "helper GPUs only with --expert-server-gpu\n");
        return 1;
    }
    if (cfg.gpu && d.remote_count > 0)
        std::fprintf(stderr, "strata net: GPU expert server: %d helper GPU(s) compute the served experts they hold; their rows are the GPU "
                             "kernels' (within kernel tolerance of the conductor's CPU rows, not bit-identical)\n", d.remote_count);
    ExpertServeCfg c2 = cfg;
    c2.identity.arch = kArchQwen38;
    c2.identity.n_embd = (uint32_t) strata::kernels::cpu::H;
    const ExpertComputeFn compute = [&d](uint32_t layer, uint32_t n_tok, uint32_t k, int arith, const int32_t* ids, const uint8_t* mask,
                                         const float* x, float* out, std::string& err) {
        d.failed = false; d.fail = nullptr;   // a failure answers one request; it must not poison the next
        d.layers = layer;
        d.only_mask = mask;
        strata::core::expert_pool_dispatch_multi(d, x, ids, n_tok, k, out,
                                                  arith == 1 ? strata::kernels::cpu::NativeArithmetic::WindowInvariant
                                                             : strata::kernels::cpu::NativeArithmetic::Default);
        d.only_mask = nullptr;
        if (d.failed) {
            err = d.fail != nullptr ? d.fail : "?";
            d.failed = false; d.fail = nullptr;
            return false;
        }
        return true;
    };
    return serve_experts(c2, compute);
}

}  // namespace strata::core
