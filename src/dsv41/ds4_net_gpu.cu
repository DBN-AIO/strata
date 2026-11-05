// Strata. The DeepSeek-V4.1-Flash GPU expert server's compute side (ds4_net_gpu.hpp, docs/DEEPSEEK41_SPLIT.md "GPU expert server").
//
// The served layers' routed experts are copied from the pack's shards into VRAM once (one allocation a layer: expert e's blob at
// base + (e - elo) * stride, gate | up | down each 256-byte aligned: ds4_gpu_place.hpp), and a request is answered with the engine's
// window kernels: the window's (token, expert) pairs become a plan [T][8] of blob addresses, glmf::moe_win_group groups them by
// expert (each expert read once for all the tokens that picked it), gguf::win_rows runs gate / up, the swiglu and down on the card,
// and the entries' un-weighted down rows come back.  The conductor's weighted sum, the shared expert and the residual are not
// here, as on the CPU server.
#include "ds4_net_gpu.hpp"

#include "ds4_gguf.cuh"
#include "ds4_os.hpp"
#include "strata/dsv41/moe_win.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::ds4::net {
namespace gg = strata::ds4::gguf;
namespace os = strata::ds4::os;
namespace x3 = strata::dsv41k::exl3;
namespace gf = strata::dsv41k::glmf;

namespace {
using Clock = std::chrono::steady_clock;
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

constexpr int kMaxEnt = kMaxTok * 8;   // the window's entries as the engine's tables hold them (t * 8 + i)

bool cu(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}
}  // namespace

struct GpuExpertHost::Impl {
    const ExpertMap* m = nullptr;
    int lo = 0, hi = -1, elo = 0, ehi = -1, ne = 0, dev = 0;
    GpuPlace place;
    std::string dev_name;
    struct Lay {
        bool on = false;
        uint8_t* base = nullptr;
        GpuBlob blob;
        x3::ExpertLayout lay;
        uint64_t bytes = 0;
    };
    std::vector<Lay> L;   // per layer of the pack
    cudaStream_t s = nullptr;
    // work buffers (one window at a time: compute() holds mu)
    float* d_x = nullptr;
    float* d_h = nullptr;
    float* d_part = nullptr;
    unsigned long long* d_plan = nullptr;
    unsigned long long* d_grp_ptr = nullptr;
    int *d_grp_start = nullptr, *d_grp_ent = nullptr, *d_n_groups = nullptr;
    float* h_x = nullptr;      // pinned
    float* h_part = nullptr;   // pinned
    std::mutex mu;
    uint64_t n_calls = 0, n_rows = 0, resident = 0;
    double t_ms = 0;

    void release() {
        cudaSetDevice(dev);
        for (Lay& l : L) if (l.base != nullptr) cudaFree(l.base);
        cudaFree(d_x); cudaFree(d_h); cudaFree(d_part); cudaFree(d_plan); cudaFree(d_grp_ptr); cudaFree(d_grp_start);
        cudaFree(d_grp_ent); cudaFree(d_n_groups);
        if (h_x != nullptr) cudaFreeHost(h_x);
        if (h_part != nullptr) cudaFreeHost(h_part);
        if (s != nullptr) cudaStreamDestroy(s);
    }
    ~Impl() { release(); }
};

GpuExpertHost::GpuExpertHost() : p_(new Impl) {}
GpuExpertHost::~GpuExpertHost() = default;
bool GpuExpertHost::holds(int layer) const { return layer >= 0 && (size_t) layer < p_->L.size() && p_->L[(size_t) layer].on; }
int GpuExpertHost::layers_on_gpu() const { int n = 0; for (const auto& l : p_->L) n += l.on ? 1 : 0; return n; }
uint64_t GpuExpertHost::resident_bytes() const { return p_->resident; }
const GpuPlace& GpuExpertHost::plan() const { return p_->place; }
std::string GpuExpertHost::device_name() const { return p_->dev_name; }
uint64_t GpuExpertHost::calls() const { return p_->n_calls; }
uint64_t GpuExpertHost::rows() const { return p_->n_rows; }
double GpuExpertHost::compute_ms() const { return p_->t_ms; }

bool GpuExpertHost::open(const ExpertMap& m, int lo, int hi, int elo, int ehi, const GpuOpts& o, std::string& err) {
    Impl& I = *p_;
    I.m = &m; I.lo = lo; I.hi = hi < 0 ? m.n_layers - 1 : hi; I.dev = o.device;
    I.elo = std::max(0, elo); I.ehi = (ehi < 0 || ehi >= m.n_exp) ? m.n_exp - 1 : ehi;
    I.ne = I.ehi - I.elo + 1;
    if (I.lo < 0 || I.hi >= m.n_layers || I.lo > I.hi || I.ne < 1) { err = "gpu expert server: served layers / experts outside the pack"; return false; }
    if (m.dim % 8 != 0 || m.dim > 8192 || m.top_k < 1 || m.top_k > 8) { err = "gpu expert server: geometry not covered by the device kernels"; return false; }
    // The CPU lane rounds a low-bit expert's activations as ggml's Q8_K (one scale a 256 values); the device does that only when
    // DS4_IQ_Q8K=1 (ds4_gguf.cu iq_q8k, opt-in).  A server's rows must be the CPU server's, so the server turns it on (an
    // explicit DS4_IQ_Q8K in the environment wins).  Without it the rows differ by 10-30 % of their RMS on random weights.
    if (std::getenv("DS4_IQ_Q8K") == nullptr) {
#if defined(_WIN32)
        _putenv_s("DS4_IQ_Q8K", "1");
#else
        setenv("DS4_IQ_Q8K", "1", 1);
#endif
    }
    if (!cu(cudaSetDevice(o.device), "cudaSetDevice", err)) return false;
    cudaDeviceProp prop;
    if (cu(cudaGetDeviceProperties(&prop, o.device), "cudaGetDeviceProperties", err)) I.dev_name = std::string(prop.name) + " sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
    if (!cu(cudaStreamCreateWithFlags(&I.s, cudaStreamNonBlocking), "cudaStreamCreate", err)) return false;
    // the kernels' own scratch first, then the window buffers, so the free VRAM the plan reads is what is really left
    if (!gg::prepare(I.s)) { err = "gpu expert server: cannot allocate the kernels' scratch"; return false; }
    const size_t D = (size_t) m.dim, FF = (size_t) m.ff;
    if (!cu(cudaMalloc(&I.d_x, (size_t) kMaxTok * D * 4), "cudaMalloc x", err) || !cu(cudaMalloc(&I.d_h, (size_t) kMaxEnt * FF * 4), "cudaMalloc h", err) ||
        !cu(cudaMalloc(&I.d_part, (size_t) kMaxEnt * D * 4), "cudaMalloc part", err) || !cu(cudaMalloc(&I.d_plan, (size_t) kMaxEnt * 8), "cudaMalloc plan", err) ||
        !cu(cudaMalloc(&I.d_grp_ptr, (size_t) kMaxEnt * 8), "cudaMalloc grp", err) || !cu(cudaMalloc(&I.d_grp_start, (size_t) (kMaxEnt + 1) * 4), "cudaMalloc grp", err) ||
        !cu(cudaMalloc(&I.d_grp_ent, (size_t) kMaxEnt * 4), "cudaMalloc grp", err) || !cu(cudaMalloc(&I.d_n_groups, 4), "cudaMalloc grp", err) ||
        !cu(cudaMallocHost(&I.h_x, (size_t) kMaxTok * D * 4), "cudaMallocHost", err) || !cu(cudaMallocHost(&I.h_part, (size_t) kMaxEnt * D * 4), "cudaMallocHost", err))
        return false;

    // what each served layer takes, and which can be on the card at all
    I.L.assign((size_t) m.n_layers, Impl::Lay());
    std::vector<int> ids;
    GpuPlaceIn pin;
    std::string unsupported;
    for (int il = I.lo; il <= I.hi; ++il) {
        GpuBlob b;
        std::string why;
        if (!gpu_blob_layout(m, il, b, why)) { err = "gpu expert server: " + why; return false; }
        const ExpertLayer& El = m.layers[(size_t) il];
        bool dev_ok = true;
        for (int r = 0; r < 3; ++r) dev_ok = dev_ok && gg::type_on_device(El.type[r]);
        for (int r = 0; r < 3; ++r) if (El.stride[r] < b.mat[r]) { err = "gpu expert server: layer " + std::to_string(il) + ": the pack's expert stride is shorter than a matrix"; return false; }
        if (!dev_ok) {
            if (!o.cpu_rest) { err = "gpu expert server: layer " + std::to_string(il) + " has an expert type the device kernels do not take (--gpu-cpu-rest leaves it to the CPU lane)"; return false; }
            unsupported += (unsupported.empty() ? "" : ",") + std::to_string(il);
            continue;
        }
        Impl::Lay& l = I.L[(size_t) il];
        l.blob = b;
        l.bytes = (uint64_t) b.stride * (uint64_t) I.ne;
        l.lay.fmt = gg::kFmtMXFP4;   // "a GGUF's experts, read in place": the roles' types in K (0 = MXFP4)
        for (int r = 0; r < 3; ++r) { l.lay.trellis[r] = b.off[r]; l.lay.K[r] = El.type[r] == gg::kTypeMXFP4 ? 0 : El.type[r]; }
        ids.push_back(il);
        pin.layer_bytes.push_back(l.bytes);
    }
    size_t fr = 0, tot = 0;
    if (!cu(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo", err)) return false;
    pin.vram_free = fr; pin.cap = o.cap_bytes; pin.reserve = o.reserve_bytes; pin.cpu_rest = o.cpu_rest;
    const GpuPlace pl = gpu_plan(pin);
    I.place = pl;
    if (!pl.ok) { err = "gpu expert server: " + pl.why; return false; }
    if (!unsupported.empty()) std::fprintf(stderr, "strata net: gpu expert server: layers %s have expert types with no device kernel: CPU lane\n", unsupported.c_str());

    // upload: matrix r of experts elo..ehi of a layer is `ne` rows of mat[r] bytes, the pack's stride[r] apart, into the blobs' slots
    std::vector<const uint8_t*> maps(m.files.size(), nullptr);
    std::vector<uint64_t> sizes(m.files.size(), 0);
    std::vector<os::fd_t> fds(m.files.size(), os::kBadFd);
    auto close_all = [&] {
        for (size_t i = 0; i < maps.size(); ++i) if (maps[i] != nullptr) os::unmap((void*) maps[i], sizes[i]);
        for (os::fd_t fd : fds) if (fd != os::kBadFd) os::close_fd(fd);
    };
    const double t0 = now_ms();
    for (size_t q = 0; q < ids.size(); ++q) {
        if (!pl.on_gpu[q]) continue;
        const int il = ids[q];
        Impl::Lay& l = I.L[(size_t) il];
        const ExpertLayer& El = m.layers[(size_t) il];
        if (!cu(cudaMalloc(&l.base, (size_t) l.bytes), "cudaMalloc experts", err)) { close_all(); return false; }
        for (int r = 0; r < 3; ++r) {
            const size_t fi = (size_t) El.file[r];
            if (maps[fi] == nullptr) {
                const std::string path = m.full_path((int) fi);
                fds[fi] = os::open_ro(path);
                if (fds[fi] == os::kBadFd) { err = path + ": cannot open"; close_all(); return false; }
                sizes[fi] = os::file_size(fds[fi]);
                void* mp = os::map_ro(fds[fi], sizes[fi]);
                if (mp == nullptr) { err = path + ": cannot map"; close_all(); return false; }
                maps[fi] = (const uint8_t*) mp;
            }
            if (El.off[r] + (uint64_t) m.n_exp * El.stride[r] > sizes[fi]) { err = "layer " + std::to_string(il) + ": the shard is shorter than the pack's tensors say"; close_all(); return false; }
            const uint8_t* src = maps[fi] + El.off[r] + (uint64_t) I.elo * El.stride[r];
            // a few experts a copy: the pageable source goes through the driver's staging buffers, so a small height keeps them small
            for (int e0 = 0; e0 < I.ne; e0 += 8) {
                const int h = std::min(8, I.ne - e0);
                if (!cu(cudaMemcpy2D(l.base + (size_t) e0 * l.blob.stride + l.blob.off[r], l.blob.stride, src + (size_t) e0 * El.stride[r], (size_t) El.stride[r],
                                     l.blob.mat[r], (size_t) h, cudaMemcpyHostToDevice), "upload", err)) { close_all(); return false; }
            }
        }
        l.on = true;
        I.resident += l.bytes;
    }
    close_all();
    std::fprintf(stderr, "strata net: gpu expert server on %s: %d of %d served layers in VRAM, %s (experts %d-%d), uploaded in %.1f s; %s\n",
                 I.dev_name.c_str(), I.place.ok ? (int) std::count(pl.on_gpu.begin(), pl.on_gpu.end(), (char) 1) : 0, (int) ids.size(), gpu_gib(I.resident).c_str(),
                 I.elo, I.ehi, (now_ms() - t0) / 1000.0, pl.why.empty() ? "everything fits" : pl.why.c_str());
    return true;
}

bool GpuExpertHost::compute(uint32_t layer, uint32_t n_tok, uint32_t k, const int32_t* ids, const uint8_t* mask, const float* x, float* out,
                            std::string& err) {
    Impl& I = *p_;
    std::lock_guard<std::mutex> lk(I.mu);
    const ExpertMap& m = *I.m;
    if (!holds((int) layer)) { err = "layer " + std::to_string(layer) + " is not in this server's VRAM"; return false; }
    if (n_tok < 1 || (int) n_tok > kMaxTok || (int) k != m.top_k) { err = "window outside the limits"; return false; }
    const double t0 = now_ms();
    const Impl::Lay& l = I.L[(size_t) layer];
    const size_t D = (size_t) m.dim;
    unsigned long long plan[kMaxEnt] = {};
    const int n_all = (int) (n_tok * k);
    int n_rows = 0;
    for (int i = 0; i < n_all; ++i) {
        if (!mask[i]) continue;
        if (ids[i] < I.elo || ids[i] > I.ehi) { err = "expert " + std::to_string(ids[i]) + " is not served here"; return false; }
        plan[(i / (int) k) * 8 + (i % (int) k)] = (unsigned long long) (uintptr_t) (l.base + (size_t) (ids[i] - I.elo) * l.blob.stride);
        ++n_rows;
    }
    if (n_rows == 0) return true;
    if (!cu(cudaSetDevice(I.dev), "cudaSetDevice", err)) return false;
    std::memcpy(I.h_x, x, (size_t) n_tok * D * 4);
    if (!cu(cudaMemcpyAsync(I.d_x, I.h_x, (size_t) n_tok * D * 4, cudaMemcpyHostToDevice, I.s), "copy x", err) ||
        !cu(cudaMemcpyAsync(I.d_plan, plan, (size_t) n_tok * 8 * 8, cudaMemcpyHostToDevice, I.s), "copy plan", err)) return false;
    gf::MoeWin mw;
    mw.plan_ptr = I.d_plan;
    mw.T = (int) n_tok;
    mw.k = (int) k;
    mw.grp_ptr = I.d_grp_ptr;
    mw.grp_start = I.d_grp_start;
    mw.grp_ent = I.d_grp_ent;
    mw.n_groups = I.d_n_groups;
    gf::moe_win_group(mw, I.s);
    x3::WinArgs wa;
    wa.grp_ptr = I.d_grp_ptr;
    wa.grp_start = I.d_grp_start;
    wa.grp_ent = I.d_grp_ent;
    wa.n_groups = I.d_n_groups;
    wa.plan_ptr = I.d_plan;
    wa.x = I.d_x;
    wa.x_stride = D;
    wa.h = I.d_h;
    wa.part = I.d_part;
    wa.lay = l.lay;
    wa.T = (int) n_tok;
    wa.k = (int) k;
    wa.n_embd = m.dim;
    wa.n_ff = m.ff;
    wa.limit = m.lim;
    if (!gg::win_rows(wa, I.s)) { err = "gpu expert server: win_rows failed"; return false; }
    // token t's k rows (entries t * 8 .. t * 8 + k - 1) as one run of k * dim floats
    if (!cu(cudaMemcpy2DAsync(I.h_part, (size_t) k * D * 4, I.d_part, (size_t) 8 * D * 4, (size_t) k * D * 4, n_tok, cudaMemcpyDeviceToHost, I.s), "copy rows", err) ||
        !cu(cudaStreamSynchronize(I.s), "sync", err)) return false;
    for (int i = 0; i < n_all; ++i)
        if (mask[i]) std::memcpy(out + (size_t) i * D, I.h_part + (size_t) i * D, D * 4);
    ++I.n_calls;
    I.n_rows += (uint64_t) n_rows;
    I.t_ms += now_ms() - t0;
    return true;
}

}  // namespace strata::ds4::net
