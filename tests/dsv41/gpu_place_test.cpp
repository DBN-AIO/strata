// Strata. The GPU expert server's fit and placement rules, no card needed (ds4_gpu_place.hpp; docs/DEEPSEEK41_SPLIT.md).
#include "ds4_gpu_place.hpp"

#include <cstdio>
#include <string>

using namespace strata::ds4::net;

static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++g_fail; } } while (0)

static constexpr uint64_t MiB = 1024ull * 1024;

int main() {
    {   // everything fits: one card holds the lot
        GpuPlaceIn in; in.layer_bytes = {100 * MiB, 100 * MiB, 100 * MiB}; in.vram_free = 1000 * MiB; in.reserve = 200 * MiB;
        const GpuPlace p = gpu_plan(in);
        CHECK(p.ok, "fits: ok");
        CHECK(p.on_gpu.size() == 3 && p.on_gpu[0] && p.on_gpu[1] && p.on_gpu[2], "fits: every layer on the GPU");
        CHECK(p.budget == 800 * MiB && p.need == 300 * MiB && p.placed == 300 * MiB && p.shortfall == 0, "fits: budget = free - reserve");
    }
    {   // a cap below the free VRAM wins; a cap above it cannot raise the budget
        GpuPlaceIn in; in.layer_bytes = {100 * MiB, 100 * MiB}; in.vram_free = 1000 * MiB; in.reserve = 100 * MiB; in.cap = 150 * MiB;
        GpuPlace p = gpu_plan(in);
        CHECK(!p.ok && p.budget == 150 * MiB && p.shortfall == 50 * MiB, "cap 150 MiB: refused, short by 50 MiB");
        CHECK(p.why.find("short by") != std::string::npos, "the refusal says 'short by'");
        CHECK(p.placed == 0 && p.on_gpu.size() == 2 && !p.on_gpu[0] && !p.on_gpu[1], "refused: nothing placed");
        in.cap = 5000 * MiB;
        p = gpu_plan(in);
        CHECK(p.ok && p.budget == 900 * MiB, "cap above the free VRAM: the free VRAM minus the reserve");
    }
    {   // an exact fit, then 50 MiB short: refused with the exact shortfall; --gpu-cpu-rest puts the first layers on the card
        GpuPlaceIn in; in.layer_bytes = {100 * MiB, 100 * MiB, 100 * MiB}; in.vram_free = 400 * MiB; in.reserve = 100 * MiB;
        GpuPlace p = gpu_plan(in);
        CHECK(p.ok && p.need == 300 * MiB && p.budget == 300 * MiB, "exact fit: budget == need is ok");
        in.vram_free = 350 * MiB;
        p = gpu_plan(in);
        CHECK(!p.ok && p.shortfall == 50 * MiB, "50 MiB short: refused with the exact shortfall");
        in.cpu_rest = true;
        p = gpu_plan(in);
        CHECK(p.ok && p.on_gpu.size() == 3 && p.on_gpu[0] && p.on_gpu[1] && !p.on_gpu[2], "cpu_rest: the first layers that fit, the last on the CPU");
        CHECK(p.placed == 200 * MiB, "cpu_rest: placed bytes");
    }
    {   // cpu_rest skips a layer that does not fit and takes a later one that does; nothing fitting is a refusal
        GpuPlaceIn in; in.layer_bytes = {300 * MiB, 100 * MiB}; in.vram_free = 250 * MiB; in.reserve = 100 * MiB; in.cpu_rest = true;
        GpuPlace p = gpu_plan(in);
        CHECK(p.ok && !p.on_gpu[0] && p.on_gpu[1], "cpu_rest: first-fit skips an oversized layer");
        in.vram_free = 150 * MiB;
        p = gpu_plan(in);
        CHECK(!p.ok && p.placed == 0, "cpu_rest, nothing fits: refused (it would be a CPU server in disguise)");
    }
    {   // free VRAM below the reserve: a budget of 0, never an underflow
        GpuPlaceIn in; in.layer_bytes = {10 * MiB}; in.vram_free = 50 * MiB; in.reserve = 100 * MiB;
        const GpuPlace p = gpu_plan(in);
        CHECK(!p.ok && p.budget == 0 && p.shortfall == 10 * MiB, "free below the reserve: budget 0, shortfall = need");
    }
    {   // no layers served: refused (a typo in --serve-layers must not start an empty GPU server)
        GpuPlaceIn in; in.vram_free = 1000 * MiB;
        CHECK(!gpu_plan(in).ok, "no layers: refused");
    }
    {   // the blob of one expert in VRAM: gate, up, down, each aligned
        ExpertMap m;
        m.dim = 5120; m.ff = 2304; m.n_layers = 1; m.n_exp = 48; m.top_k = 6;
        m.layers.resize(1);
        m.layers[0].il = 0;
        m.layers[0].type[0] = 16; m.layers[0].type[1] = 16; m.layers[0].type[2] = 29;   // IQ2_XXS, IQ2_XXS, IQ1_M
        GpuBlob b; std::string why;
        CHECK(gpu_blob_layout(m, 0, b, why), "layout: IQ2_XXS / IQ1_M known");
        CHECK(b.mat[0] == (size_t) 2304 * (5120 / 256) * 66 && b.mat[1] == b.mat[0], "layout: gate and up bytes");
        CHECK(b.mat[2] == (size_t) 5120 * (2304 / 256) * 56, "layout: down bytes (IQ1_M, 56 B a block)");
        CHECK(b.off[0] == 0 && b.off[1] % kGpuAlign == 0 && b.off[2] % kGpuAlign == 0 && b.stride % kGpuAlign == 0, "layout: aligned");
        CHECK(b.off[1] >= b.mat[0] && b.off[2] >= b.off[1] + b.mat[1] && b.stride >= b.off[2] + b.mat[2], "layout: no overlap");
        m.layers[0].type[2] = 999;
        CHECK(!gpu_blob_layout(m, 0, b, why) && !why.empty(), "layout: an unknown type is refused with a reason");
        m.layers[0].type[2] = 29; m.layers[0].il = -1;
        CHECK(!gpu_blob_layout(m, 0, b, why), "layout: a layer the pack has no experts for is refused");
    }
    if (g_fail == 0) std::printf("gpu_place_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
