// Strata. BBXL S7: the chunked prompt flow through the expert servers - the host-side arithmetic, header only, no CUDA.
//
// The engine's chunked flow (forward_chunks) takes a chunk of T tokens through a layer.  For a served layer the experts a server owns
// are not streamed from this PC's files: the chunk's FFN input rows and the router's picks go to net::Remote::prompt in slabs, and the
// un-weighted expert rows that come back are summed into the layer's accumulator with the SAME kernel the local experts use (moe_combine:
// acc = fma(w, row, acc), a token's picks in ascending order).  This file builds the slab's inputs (which picks are sent, where each row
// sits in the answer, the weights) and holds a CPU copy of that kernel, so the engine and tests/dsv41/ds4_net_test.cpp (--chunk-flow) run
// one piece of code and the synthetic test can check the sums bit for bit without a GPU.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4::net {

struct ChunkSlab {
    int t0 = 0, n = 0, k = 0, stride = 0;
    std::vector<int32_t> ids;   ///< n x k: the router's expert id of a pick a server owns, -1 for the rest (a -1 is never sent)
    std::vector<int32_t> pos;   ///< n x stride: pick i of token t -> its row in the answer (t * k + i), -1 where nothing was sent
    std::vector<float> w;       ///< n x stride: the route weight of a sent pick, 0 elsewhere
    size_t sent = 0;            ///< picks sent
};

/// ridx / rw: the chunk's routing on the host (ridx[t * stride + i] expert id, rw[t * stride + i] weight; the engine's kPicks stride);
/// tokens [t0, t0 + n) of it.  owned(e): this expert is a server's (and the conductor does not run it itself).
template <class Owned>
inline void chunk_slab_build(ChunkSlab& s, const int* ridx, const float* rw, int stride, int k, int t0, int n, Owned owned) {
    s.t0 = t0; s.n = n; s.k = k; s.stride = stride; s.sent = 0;
    s.ids.assign((size_t) n * (size_t) k, -1);
    s.pos.assign((size_t) n * (size_t) stride, -1);
    s.w.assign((size_t) n * (size_t) stride, 0.0f);
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < k; ++i) {
            const int e = ridx[(size_t) (t0 + t) * (size_t) stride + (size_t) i];
            if (e < 0 || !owned(e)) continue;
            s.ids[(size_t) t * (size_t) k + (size_t) i] = e;
            s.pos[(size_t) t * (size_t) stride + (size_t) i] = t * k + i;
            s.w[(size_t) t * (size_t) stride + (size_t) i] = rw[(size_t) (t0 + t) * (size_t) stride + (size_t) i];
            ++s.sent;
        }
}

/// The engine's moe_combine kernel on the CPU (ds4_kernels.cu, r0 = 0, r1 = n * k): acc (n x dim, ld apart) += the slab's weighted rows.
inline void chunk_combine_ref(float* acc, size_t ld, const float* rows, const ChunkSlab& s, int dim) {
    for (int t = 0; t < s.n; ++t)
        for (int e = 0; e < dim; ++e) {
            float a = acc[(size_t) t * ld + (size_t) e];
            bool any = false;
            for (int i = 0; i < s.k; ++i) {
                const int r = s.pos[(size_t) t * (size_t) s.stride + (size_t) i];
                if (r < 0 || r >= s.n * s.k) continue;
                a = std::fmaf(s.w[(size_t) t * (size_t) s.stride + (size_t) i], rows[(size_t) r * (size_t) dim + (size_t) e], a);
                any = true;
            }
            if (any) acc[(size_t) t * ld + (size_t) e] = a;
        }
}

}  // namespace strata::ds4::net
