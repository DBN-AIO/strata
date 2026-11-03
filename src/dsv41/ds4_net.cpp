// Strata. DeepSeek-V4.1-Flash experts over the LAN: the expert map, the CPU expert host, the conductor's Remote and the
// server main.  Host only (no CUDA).  See ds4_net.hpp and docs/REMOTE_EXPERTS.md "DeepSeek-V4.1-Flash".
#include "ds4_net.hpp"

#include "ds4_gguf_cpu.hpp"
#include "ds4_os.hpp"
#include "strata/core/net_proto.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace strata::ds4::net {
namespace gg = strata::ds4::gguf;
namespace np = strata::core::netproto;
namespace os = strata::ds4::os;

namespace {
using Clock = std::chrono::steady_clock;
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

constexpr int kTypeMXFP4 = 39;
struct TypeInfo { const char* name; int id; int vals; int bytes; };
// ggml's block sizes (ggml-common.h): values and bytes of one block
const TypeInfo kTypes[] = {{"MXFP4", 39, 32, 17},   {"Q2_K", 10, 256, 84},    {"Q3_K", 11, 256, 110},   {"Q4_K", 12, 256, 144},
                           {"Q5_K", 13, 256, 176},  {"Q6_K", 14, 256, 210},   {"IQ2_XXS", 16, 256, 66}, {"IQ2_XS", 17, 256, 74},
                           {"IQ3_XXS", 18, 256, 98}, {"IQ1_S", 19, 256, 50},  {"IQ4_NL", 20, 32, 18},   {"IQ3_S", 21, 256, 110},
                           {"IQ2_S", 22, 256, 82},  {"IQ4_XS", 23, 256, 136}, {"IQ1_M", 29, 256, 56}};
const TypeInfo* info(int id) {
    for (const TypeInfo& t : kTypes) if (t.id == id) return &t;
    return nullptr;
}

size_t round64(size_t n) { return (n + 63) & ~(size_t) 63; }

// a worker pool: parallel_for(n, f) runs f(0..n-1) on the pool and the calling thread
class Pool {
public:
    explicit Pool(int n) {
        for (int i = 1; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }
    void parallel_for(int n, const std::function<void(int)>& f) {
        if (n <= 0) return;
        if (n == 1 || th_.empty()) { for (int j = 0; j < n; ++j) f(j); return; }
        std::unique_lock<std::mutex> lk(m_);
        f_ = &f; n_ = n; next_ = 0; ++gen_;
        lk.unlock();
        cv_.notify_all();
        work();
        lk.lock();
        done_.wait(lk, [&] { return busy_ == 0; });
        f_ = nullptr;
    }
private:
    void work() {
        for (;;) {
            const int j = next_.fetch_add(1);
            if (j >= n_) break;
            (*f_)(j);
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            ++busy_;
            lk.unlock();
            work();
            lk.lock();
            if (--busy_ == 0) done_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)>* f_ = nullptr;
    std::atomic<int> next_{0};
    int n_ = 0, busy_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

struct Aligned {   // a 64-byte aligned scratch
    std::vector<uint8_t> raw;
    uint8_t* p = nullptr;
    void resize(size_t n) {
        if (raw.size() < n + 64) raw.resize(n + 64);
        p = (uint8_t*) (((uintptr_t) raw.data() + 63) & ~(uintptr_t) 63);
    }
};
}  // namespace

int type_of_name(const std::string& dtype) {
    for (const TypeInfo& t : kTypes) if (dtype == t.name) return t.id;
    return 0;
}
size_t type_row_bytes(int type, int64_t k) {
    const TypeInfo* t = info(type);
    return t != nullptr && k > 0 && k % t->vals == 0 ? (size_t) (k / t->vals) * (size_t) t->bytes : 0;
}

// ================================================================== the expert map
bool ExpertMap::load(const std::string& dir, std::string& err) {
    pack_dir = dir;
    std::unordered_map<std::string, std::string> kv;
    {
        std::ifstream f(dir + "/ds4.txt");
        if (!f) { err = dir + "/ds4.txt: not found"; return false; }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    auto num = [&](const char* k, double d) { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); };
    dim = (int) num("hidden_size", 0); ff = (int) num("moe_intermediate_size", 0); n_layers = (int) num("num_hidden_layers", 0);
    n_exp = (int) num("n_routed_experts", 0); top_k = (int) num("num_experts_per_tok", 0); lim = (float) num("swiglu_limit", 0.0);
    if (kv.count("model_dir") == 0) { err = dir + "/ds4.txt: no model_dir"; return false; }
    model_dir = kv["model_dir"];
    if (dim <= 0 || ff <= 0 || n_layers <= 0 || n_exp <= 0 || top_k <= 0 || top_k > kMaxPicks) {
        err = dir + "/ds4.txt: incomplete or uncovered geometry (hidden_size, moe_intermediate_size, num_hidden_layers, n_routed_experts, num_experts_per_tok <= " +
              std::to_string(kMaxPicks) + ")";
        return false;
    }
    struct T { int type = 0, file = -1; uint64_t off = 0, bytes = 0; bool seen = false; };
    // layer -> matrix -> expert
    std::vector<std::vector<std::vector<T>>> t((size_t) n_layers, std::vector<std::vector<T>>(3, std::vector<T>((size_t) n_exp)));
    std::vector<std::vector<std::string>> dtype_of((size_t) n_layers, std::vector<std::string>(3));
    std::ifstream f(dir + "/tensors.txt");
    if (!f) { err = dir + "/tensors.txt: not found"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 3) continue;
        if (line[0] == 'f') {
            std::istringstream ss(line);
            std::string tag, name; int i = 0;
            if ((ss >> tag >> i >> name) && tag == "file" && i >= 0) {
                if ((int) files.size() <= i) files.resize((size_t) i + 1);
                files[(size_t) i] = name;
            }
            continue;
        }
        if (line[0] != 't') continue;
        // t layers.N.ffn.experts.E.wR.weight DTYPE file off bytes dims...
        const size_t a = line.find(' ');
        const size_t b = line.find(' ', a + 1);
        if (a == std::string::npos || b == std::string::npos) continue;
        const std::string name = line.substr(a + 1, b - a - 1);
        if (name.compare(0, 7, "layers.") != 0 || name.find(".ffn.experts.") == std::string::npos) continue;
        int il = -1, e = -1, w = -1;
        if (std::sscanf(name.c_str(), "layers.%d.ffn.experts.%d.w%d.weight", &il, &e, &w) != 3) continue;
        if (il < 0 || il >= n_layers || e < 0 || e >= n_exp || (w != 1 && w != 2 && w != 3)) continue;
        std::istringstream ss(line.substr(b + 1));
        std::string dt; int fi = 0; unsigned long long off = 0, nb = 0;
        if (!(ss >> dt >> fi >> off >> nb)) continue;
        const int r = w == 1 ? 0 : w == 3 ? 1 : 2;   // gate, up, down
        T& x = t[(size_t) il][(size_t) r][(size_t) e];
        x.type = type_of_name(dt); x.file = fi; x.off = off; x.bytes = nb; x.seen = true;
        dtype_of[(size_t) il][(size_t) r] = dt;
    }
    layers.assign((size_t) n_layers, ExpertLayer());
    for (int il = 0; il < n_layers; ++il) {
        if (!t[(size_t) il][0][0].seen) continue;   // a layer without routed experts
        ExpertLayer& L = layers[(size_t) il];
        L.il = il;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? ff : dim, kk = r < 2 ? dim : ff;
            const T& t0 = t[(size_t) il][(size_t) r][0];
            const T& t1 = n_exp > 1 ? t[(size_t) il][(size_t) r][1] : t0;
            const std::string where = "layers." + std::to_string(il) + ".ffn.experts.*.w" + (r == 0 ? "1" : r == 1 ? "3" : "2");
            if (t0.type == 0) { err = where + ": type " + dtype_of[(size_t) il][(size_t) r] + " is not one the CPU lane computes"; return false; }
            const size_t want = (size_t) n * type_row_bytes(t0.type, kk);
            if (want == 0 || t0.bytes != want) { err = where + ": " + std::to_string(t0.bytes) + " bytes, expected " + std::to_string(want); return false; }
            L.type[r] = t0.type; L.file[r] = t0.file; L.off[r] = t0.off; L.bytes[r] = t0.bytes;
            L.stride[r] = n_exp > 1 ? t1.off - t0.off : t0.bytes;
            for (int e = 0; e < n_exp; ++e) {
                const T& x = t[(size_t) il][(size_t) r][(size_t) e];
                if (!x.seen || x.type != t0.type || x.bytes != t0.bytes || x.file != t0.file || x.off != t0.off + (uint64_t) e * L.stride[r]) {
                    err = where + ": expert " + std::to_string(e) + " missing, or not where the layer's others are";
                    return false;
                }
            }
        }
        if (L.type[0] != L.type[1]) { err = "layer " + std::to_string(il) + ": gate and up must have one type"; return false; }
        const bool mx = L.type[0] == kTypeMXFP4 && L.type[1] == kTypeMXFP4 && L.type[2] == kTypeMXFP4;
        if (!mx && (L.type[0] == kTypeMXFP4 || L.type[2] == kTypeMXFP4)) { err = "layer " + std::to_string(il) + ": MXFP4 does not mix with the other types"; return false; }
    }
    file_sizes.assign(files.size(), 0);
    for (size_t i = 0; i < files.size(); ++i) {
        std::FILE* fp = std::fopen(full_path((int) i).c_str(), "rb");
        if (fp == nullptr) continue;
#if defined(_WIN32)
        _fseeki64(fp, 0, SEEK_END);
        file_sizes[i] = (uint64_t) _ftelli64(fp);
#else
        fseeko(fp, 0, SEEK_END);
        file_sizes[i] = (uint64_t) ftello(fp);
#endif
        std::fclose(fp);
    }
    return true;
}

core::NetIdentity ExpertMap::identity() const {
    np::Sha256 h;
    auto u64 = [&](uint64_t v) { h.update(&v, 8); };
    h.update("DS4XP1", 6);
    u64((uint64_t) dim); u64((uint64_t) ff); u64((uint64_t) n_layers); u64((uint64_t) n_exp); u64((uint64_t) top_k);
    uint32_t lb = 0; std::memcpy(&lb, &lim, 4); u64(lb);
    for (const ExpertLayer& L : layers) {
        u64((uint64_t) (int64_t) L.il);
        if (L.il < 0) continue;
        for (int r = 0; r < 3; ++r) { u64((uint64_t) L.type[r]); u64((uint64_t) (int64_t) L.file[r]); u64(L.off[r]); u64(L.stride[r]); u64(L.bytes[r]); }
    }
    u64(files.size());
    for (size_t i = 0; i < files.size(); ++i) { u64(files[i].size()); h.update(files[i].data(), files[i].size()); u64(file_sizes[i]); }
    core::NetIdentity id;
    h.finish(id.fingerprint);
    id.isa = 0;
    id.arch = core::kArchDsv41;
    id.n_embd = (uint32_t) dim;
    return id;
}

// ================================================================== the expert host
struct ExpertHost::Impl {
    const ExpertMap* m = nullptr;
    int lo = 0, hi = -1;
    std::vector<os::fd_t> fds;
    std::vector<const uint8_t*> maps;
    std::vector<uint64_t> sizes;
    std::vector<gg::IqCpu> fm;     // per layer (iq layers)
    std::vector<char> mx;          // per layer
    std::unique_ptr<Pool> pool;
    std::mutex mu;
    Aligned xa, ha;
    std::vector<float> hf;
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> t_us{0};
    ~Impl() {
        for (size_t i = 0; i < maps.size(); ++i) if (maps[i] != nullptr) os::unmap((void*) maps[i], sizes[i]);
        for (os::fd_t fd : fds) if (fd != os::kBadFd) os::close_fd(fd);
    }
    const uint8_t* mat(const ExpertLayer& L, int e, int r) const {
        return maps[(size_t) L.file[r]] + L.off[r] + (uint64_t) e * L.stride[r];
    }
};

ExpertHost::ExpertHost() : p_(new Impl) {}
ExpertHost::~ExpertHost() = default;
uint64_t ExpertHost::calls() const { return p_->n_calls.load(); }
double ExpertHost::compute_ms() const { return (double) p_->t_us.load() / 1000.0; }

bool ExpertHost::open(const ExpertMap& m, int lo, int hi, int threads, std::string& err) {
    Impl& I = *p_;
    I.m = &m; I.lo = lo; I.hi = hi < 0 ? m.n_layers - 1 : hi;
    if (I.lo < 0 || I.hi >= m.n_layers || I.lo > I.hi) { err = "served layers " + std::to_string(lo) + "-" + std::to_string(hi) + " outside 0-" + std::to_string(m.n_layers - 1); return false; }
    I.fds.assign(m.files.size(), os::kBadFd);
    I.maps.assign(m.files.size(), nullptr);
    I.sizes.assign(m.files.size(), 0);
    I.fm.assign((size_t) m.n_layers, gg::IqCpu());
    I.mx.assign((size_t) m.n_layers, 0);
    for (int il = I.lo; il <= I.hi; ++il) {
        const ExpertLayer& L = m.layers[(size_t) il];
        if (L.il < 0) { err = "layer " + std::to_string(il) + " has no routed experts in the pack"; return false; }
        for (int r = 0; r < 3; ++r) {
            const size_t fi = (size_t) L.file[r];
            if (fi >= m.files.size()) { err = "layer " + std::to_string(il) + ": bad file index"; return false; }
            if (I.maps[fi] != nullptr) continue;
            const std::string path = m.full_path((int) fi);
            I.fds[fi] = os::open_ro(path);
            if (I.fds[fi] == os::kBadFd) { err = path + ": cannot open"; return false; }
            I.sizes[fi] = os::file_size(I.fds[fi]);
            void* p = os::map_ro(I.fds[fi], I.sizes[fi]);
            if (p == nullptr) { err = path + ": cannot map"; return false; }
            I.maps[fi] = (const uint8_t*) p;
            if (L.off[r] + (uint64_t) m.n_exp * L.stride[r] > I.sizes[fi]) { err = path + ": shorter than the pack's tensors say"; return false; }
        }
        const bool mx = L.type[0] == kTypeMXFP4 && L.type[2] == kTypeMXFP4;
        I.mx[(size_t) il] = mx ? 1 : 0;
        if (!mx) {
            std::string ferr;
            gg::IqCpu f;
            if (!gg::iq_cpu_fmt(L.type[0], L.type[2], m.dim, m.ff, f, ferr) || f.act_bytes > (size_t) m.dim * 4 || f.h_bytes > (size_t) m.dim * 4) {
                err = "layer " + std::to_string(il) + ": the CPU lane cannot compute these experts: " + (ferr.empty() ? "activation rows too large" : ferr);
                return false;
            }
            I.fm[(size_t) il] = f;
        }
    }
    int nt = threads > 0 ? threads : (int) std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;
    I.pool.reset(new Pool(nt));
    return true;
}

uint64_t ExpertHost::warm(int lo, int hi, int elo, int ehi) {
    Impl& I = *p_;
    const ExpertMap& m = *I.m;
    if (ehi < 0 || ehi >= m.n_exp) ehi = m.n_exp - 1;
    if (elo < 0) elo = 0;
    std::atomic<uint64_t> total{0};
    std::vector<std::pair<const uint8_t*, uint64_t>> jobs;
    for (int il = lo; il <= hi; ++il) {
        const ExpertLayer& L = m.layers[(size_t) il];
        if (L.il < 0) continue;
        for (int r = 0; r < 3; ++r) jobs.push_back({I.maps[(size_t) L.file[r]] + L.off[r] + L.stride[r] * (uint64_t) elo, L.stride[r] * (uint64_t) (ehi - elo + 1)});
    }
    I.pool->parallel_for((int) jobs.size(), [&](int j) {
        volatile uint8_t sink = 0;
        for (uint64_t o = 0; o < jobs[(size_t) j].second; o += 4096) sink = (uint8_t) (sink + jobs[(size_t) j].first[o]);
        (void) sink;
        total += jobs[(size_t) j].second;
    });
    return total.load();
}

bool ExpertHost::compute(uint32_t layer, uint32_t n_tok, uint32_t k, const int32_t* ids, const uint8_t* mask, const float* x, float* out,
                         std::string& err) {
    Impl& I = *p_;
    std::lock_guard<std::mutex> lk(I.mu);
    const ExpertMap& m = *I.m;
    if ((int) layer < I.lo || (int) layer > I.hi) { err = "layer " + std::to_string(layer) + " is not served here"; return false; }
    if (n_tok < 1 || (int) n_tok > kMaxTok || (int) k != m.top_k) { err = "window outside the limits"; return false; }
    const double t0 = now_ms();
    const ExpertLayer& L = m.layers[(size_t) layer];
    const int E = m.dim, FF = m.ff;
    const bool mx = I.mx[(size_t) layer] != 0;
    const gg::IqCpu& fm = I.fm[(size_t) layer];
    const float lim = m.lim;
    // entries -> jobs (one expert, up to kEntTok entries), the experts in id order
    struct Job { int e = 0, n = 0; int ent[kEntTok] = {}, tok[kEntTok] = {}, slot[kEntTok] = {}; };
    std::map<int, std::vector<int>> by;
    const int n_all = (int) (n_tok * k);
    for (int i = 0; i < n_all; ++i) if (mask[i]) by[ids[i]].push_back(i);
    std::vector<Job> jobs;
    int n_ent = 0;
    for (auto& kvp : by)
        for (size_t a = 0; a < kvp.second.size(); a += kEntTok) {
            Job j; j.e = kvp.first;
            j.n = (int) std::min<size_t>(kEntTok, kvp.second.size() - a);
            for (int q = 0; q < j.n; ++q) { j.ent[q] = kvp.second[a + (size_t) q]; j.tok[q] = j.ent[q] / (int) k; j.slot[q] = n_ent++; }
            jobs.push_back(j);
        }
    if (jobs.empty()) return true;
    const size_t xs = round64(mx ? gg::mx_act_bytes(E) : fm.act_bytes), hs = round64(mx ? gg::mx_act_bytes(FF) : fm.h_bytes);
    I.xa.resize(xs * (size_t) n_tok);
    I.ha.resize(hs * (size_t) n_ent);
    I.hf.assign((size_t) n_ent * (size_t) FF, 0.0f);
    bool need[kMaxTok] = {};
    for (const Job& j : jobs) for (int q = 0; q < j.n; ++q) need[j.tok[q]] = true;
    for (uint32_t t = 0; t < n_tok; ++t)
        if (need[t]) {
            if (mx) gg::mx_quant(x + (size_t) t * (size_t) E, E, I.xa.p + (size_t) t * xs);
            else gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * (size_t) E, E, I.xa.p + (size_t) t * xs);
        }
    constexpr int RB = 64;   // rows a task (whole blocks of h: RB % 32 == 0)
    const int GB = (FF + RB - 1) / RB, DB = (E + RB - 1) / RB, nj = (int) jobs.size();
    const size_t gr = type_row_bytes(L.type[0], E), dr = type_row_bytes(L.type[2], FF);
    // gate / up -> h
    I.pool->parallel_for(nj * GB, [&](int j) {
        const Job& jb = jobs[(size_t) (j / GB)];
        const int r0 = (j % GB) * RB, r1 = std::min(FF, r0 + RB);
        const uint8_t* gate = I.mat(L, jb.e, 0);
        const uint8_t* up = I.mat(L, jb.e, 1);
        if (mx) {
            const void* ap[kEntTok];
            float yg[kEntTok][RB], yu[kEntTok][RB];
            float *gp[kEntTok], *upp[kEntTok];
            for (int i = 0; i < jb.n; ++i) { ap[i] = I.xa.p + (size_t) jb.tok[i] * xs; gp[i] = yg[i]; upp[i] = yu[i]; }
            gg::mx_rows(gate, gr, E, r0, r1, ap, jb.n, gp);
            gg::mx_rows(up, gr, E, r0, r1, ap, jb.n, upp);
            for (int i = 0; i < jb.n; ++i) {
                float* h = I.hf.data() + (size_t) jb.slot[i] * (size_t) FF;
                for (int r = r0; r < r1; ++r) {
                    float gv = yg[i][r - r0], uv = yu[i][r - r0];
                    if (lim > 0.0f) { gv = std::min(gv, lim); uv = std::min(std::max(uv, -lim), lim); }
                    h[r] = gv / (1.0f + std::exp(-gv)) * uv;
                }
            }
        } else {
            const void* ap[kEntTok];
            float* hp[kEntTok];
            for (int i = 0; i < jb.n; ++i) { ap[i] = I.xa.p + (size_t) jb.tok[i] * xs; hp[i] = I.hf.data() + (size_t) jb.slot[i] * (size_t) FF; }
            gg::iq_cpu_gate_up(fm, gate, up, ap, jb.n, hp, r0, r1, lim);
        }
    });
    // h -> the down matrix's activation type
    // (the i-quant activation quantizer is called from one thread at a time, as the engine's lane does)
    if (mx)
        I.pool->parallel_for(n_ent, [&](int s) { gg::mx_quant(I.hf.data() + (size_t) s * (size_t) FF, FF, I.ha.p + (size_t) s * hs); });
    else
        for (int s = 0; s < n_ent; ++s) gg::iq_cpu_quant(fm.d_act, I.hf.data() + (size_t) s * (size_t) FF, FF, I.ha.p + (size_t) s * hs);
    // down -> out rows
    I.pool->parallel_for(nj * DB, [&](int j) {
        const Job& jb = jobs[(size_t) (j / DB)];
        const int r0 = (j % DB) * RB, r1 = std::min(E, r0 + RB);
        const uint8_t* down = I.mat(L, jb.e, 2);
        const void* hp[kEntTok];
        float* yp[kEntTok];
        // (mx_rows writes y[i][r - r0]: the band's pointer is the row's + r0; iq_cpu_down adds r0 itself)
        for (int i = 0; i < jb.n; ++i) { hp[i] = I.ha.p + (size_t) jb.slot[i] * hs; yp[i] = out + (size_t) jb.ent[i] * (size_t) E + (mx ? r0 : 0); }
        if (mx) gg::mx_rows(down, dr, FF, r0, r1, hp, jb.n, yp);
        else gg::iq_cpu_down(fm, down, hp, jb.n, yp, r0, r1);
    });
    I.n_calls++;
    I.t_us += (uint64_t) ((now_ms() - t0) * 1000.0);
    return true;
}

// ================================================================== the conductor's side
struct Remote::Impl {
    core::NetExperts net;
    int dim = 0, top_k = 0;
    uint64_t n_remote = 0, n_local = 0, n_fail = 0;
    std::vector<float> xc, rows;
    std::vector<int32_t> ids, kind;
};
Remote::Remote() : p_(new Impl) {}
Remote::~Remote() = default;
bool Remote::active() const { return p_->net.active(); }
bool Remote::serves(int layer) const { return p_->net.serves(layer); }
uint64_t Remote::rows_remote() const { return p_->n_remote; }
uint64_t Remote::rows_local() const { return p_->n_local; }
std::string Remote::stats() const {
    return "remote expert rows " + std::to_string(p_->n_remote) + ", computed locally " + std::to_string(p_->n_local) + ", failed calls " + std::to_string(p_->n_fail);
}
bool Remote::open(const ExpertMap& m, const std::string& spec, std::string& err, uint32_t arch_override) {
    p_->dim = m.dim; p_->top_k = m.top_k;
    core::NetIdentity id = m.identity();
    if (arch_override != 0xffffffffu) id.arch = arch_override;
    return p_->net.open(spec, m.n_layers, m.n_exp, id, err);
}
bool Remote::run(int layer, const float* x, size_t ldx, const Item* items, int n_items, float* y, std::vector<char>& remote) {
    Impl& I = *p_;
    remote.assign((size_t) n_items, 0);
    if (!I.net.serves(layer)) return true;
    const int k = I.top_k, dim = I.dim;
    // the window's distinct tokens, compacted; each token's pairs in the order they come
    int map_tok[kMaxTok];
    for (int& v : map_tok) v = -1;
    int nt = 0;
    int count[kMaxTok] = {};
    for (int q = 0; q < n_items; ++q)
        for (int i = 0; i < items[q].n; ++i) {
            const int t = items[q].tok[i];
            if (t < 0 || t >= kMaxTok) return true;   // a token past the window (none on the decode / chunk paths): local
            if (map_tok[t] < 0) map_tok[t] = nt++;
            if (++count[map_tok[t]] > k) return true; // more picks than the wire's k: local
        }
    if (nt == 0) return true;
    I.xc.resize((size_t) nt * (size_t) dim);
    for (int t = 0; t < kMaxTok; ++t)
        if (map_tok[t] >= 0) std::memcpy(I.xc.data() + (size_t) map_tok[t] * (size_t) dim, x + (size_t) t * ldx, (size_t) dim * 4);
    I.ids.assign((size_t) nt * (size_t) k, 0);
    I.kind.assign((size_t) nt * (size_t) k, -2);
    int fill[kMaxTok] = {};
    std::vector<int> pos((size_t) n_items * kMaxTok, -1);
    for (int q = 0; q < n_items; ++q)
        for (int i = 0; i < items[q].n; ++i) {
            const int ct = map_tok[items[q].tok[i]];
            const int at = ct * k + fill[ct]++;
            I.ids[(size_t) at] = items[q].e;
            I.kind[(size_t) at] = -1;
            pos[(size_t) q * kMaxTok + (size_t) i] = at;
        }
    std::string err;
    I.net.begin(layer, I.xc.data(), I.ids.data(), nt, k, I.kind.data(), 0, err);
    if (!I.net.in_flight()) { for (int q = 0; q < n_items; ++q) I.n_local += (uint64_t) items[q].n; return true; }
    I.rows.resize((size_t) nt * (size_t) k * (size_t) dim);
    const bool ok = I.net.finish(I.rows.data(), err);
    if (!ok) {
        ++I.n_fail;
        for (int32_t at : I.net.owned()) I.kind[(size_t) at] = -1;   // the hosts that failed: their rows are the caller's
    }
    for (int q = 0; q < n_items; ++q) {
        bool all = true;
        for (int i = 0; i < items[q].n; ++i) all = all && I.kind[(size_t) pos[(size_t) q * kMaxTok + (size_t) i]] == 3;
        if (!all) { I.n_local += (uint64_t) items[q].n; continue; }
        for (int i = 0; i < items[q].n; ++i)
            std::memcpy(y + (size_t) (items[q].slot0 + i) * (size_t) dim, I.rows.data() + (size_t) pos[(size_t) q * kMaxTok + (size_t) i] * (size_t) dim, (size_t) dim * 4);
        remote[(size_t) q] = 1;
        I.n_remote += (uint64_t) items[q].n;
    }
    return true;
}

// ================================================================== the server
int server_main(int argc, char** argv) {
    std::string pack, host = "127.0.0.1", layers, experts;
    int port = 7840, threads = 0;
    bool warm = true, once = false;
    static const char* const with_value[] = {"--gpus", "--max-context", "--reserve-mib", "--vram-experts", "--expert-profile", "--ssd-cache",
                                             "--ssd-policy", "--dspark", "--dspark-conf", "--experts", "--native", "--spec", "--prefill",
                                             "--expert-cache", "--pcie-frac", "--gpu"};
    static const char* const bare[] = {"--expert-server", "--mmap-experts", "--greedy", "--profile-readonly"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-dsv41 --expert-server: %s needs a value\n", w); std::exit(2); }
            return argv[++i];
        };
        if (a == "--pack") pack = val("--pack");
        else if (a == "--listen") {
            std::string why;
            if (!np::parse_hostport(val("--listen"), "127.0.0.1", 7840, true, host, port, why)) { std::fprintf(stderr, "--listen: %s\n", why.c_str()); return 2; }
        }
        else if (a == "--serve-layers") layers = val("--serve-layers");
        else if (a == "--serve-experts") experts = val("--serve-experts");
        else if (a == "--cpu-threads") threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--no-warm") warm = false;
        else if (a == "--once") once = true;
        else if (a == "--help" || a == "-h") {
            std::printf("strata-dsv41 --expert-server --pack DIR --listen [HOST:]PORT [--serve-layers A-B] [--serve-experts E0-E1] [--cpu-threads N] [--no-warm]\n"
                        "  serves the pack's routed experts of layers A-B (default: all) to a conductor started with\n"
                        "  --remote-experts HOST:PORT=A-B.  CPU only: CUDA is never touched.  STRATA_EXPERT_TOKEN sets the shared secret.\n");
            return 0;
        } else {
            bool known = false;
            for (const char* f : with_value) if (a == f) { (void) val(f); known = true; }
            for (const char* f : bare) if (a == f) known = true;
            if (!known) { std::fprintf(stderr, "strata-dsv41 --expert-server: unknown option %s\n", a.c_str()); return 2; }
        }
    }
    if (pack.empty()) { std::fprintf(stderr, "strata-dsv41 --expert-server: --pack DIR is required\n"); return 2; }
    ExpertMap map;
    std::string err;
    if (!map.load(pack, err)) { std::fprintf(stderr, "strata-dsv41 --expert-server: %s\n", err.c_str()); return 1; }
    int lo = 0, hi = map.n_layers - 1;
    if (!layers.empty()) {
        if (std::sscanf(layers.c_str(), "%d-%d", &lo, &hi) != 2) { std::fprintf(stderr, "--serve-layers needs A-B\n"); return 2; }
    }
    int elo = 0, ehi = map.n_exp - 1;   // a layer's experts may be split over hosts: --remote-experts HOST=lo-hi/e0-e1 on the conductor
    if (!experts.empty() && (std::sscanf(experts.c_str(), "%d-%d", &elo, &ehi) != 2 || elo < 0 || ehi < elo || ehi >= map.n_exp)) {
        std::fprintf(stderr, "--serve-experts needs A-B inside 0-%d\n", map.n_exp - 1);
        return 2;
    }
    ExpertHost host_;
    if (!host_.open(map, lo, hi, threads, err)) { std::fprintf(stderr, "strata-dsv41 --expert-server: %s\n", err.c_str()); return 1; }
    if (warm) {
        const double t0 = now_ms();
        const uint64_t b = host_.warm(lo, hi, elo, ehi);
        std::fprintf(stderr, "strata net: dsv41 served layers %d-%d resident %.2f GiB (touched in %.1f s)\n", lo, hi, (double) b / (1024.0 * 1024 * 1024), (now_ms() - t0) / 1000.0);
    }
    core::ExpertServeCfg cfg;
    cfg.host = host; cfg.port = port; cfg.layer_lo = lo; cfg.layer_hi = hi;
    cfg.n_layers = map.n_layers; cfg.n_expert = map.n_exp; cfg.top_k = map.top_k;
    cfg.identity = map.identity();
    cfg.max_tok = kMaxTok; cfg.max_entries = kMaxEntries; cfg.once = once; cfg.expert_lo = elo; cfg.expert_hi = ehi;
    long die_after = 0;   // test hook: the server exits after N requests, as if the host had gone (STRATA_DS4_SERVER_DIE_AFTER)
    if (const char* v = std::getenv("STRATA_DS4_SERVER_DIE_AFTER")) die_after = std::atol(v);
    std::atomic<long> served{0};
    std::fprintf(stderr, "strata net: dsv41 expert server, %d layers x %d experts (top-%d), hidden %d, ffn %d, threads %d, fingerprint %s\n",
                 map.n_layers, map.n_exp, map.top_k, map.dim, map.ff, std::max(1, threads > 0 ? threads : (int) std::thread::hardware_concurrency()),
                 np::hex(cfg.identity.fingerprint, 8).c_str());
    return core::serve_experts(cfg, [&](uint32_t layer, uint32_t n_tok, uint32_t k, int, const int32_t* ids, const uint8_t* mask, const float* x, float* out, std::string& e2) {
        if (die_after > 0 && ++served > die_after) { std::fprintf(stderr, "strata net: STRATA_DS4_SERVER_DIE_AFTER reached, exiting\n"); std::fflush(stderr); std::_Exit(3); }
        return host_.compute(layer, n_tok, k, ids, mask, x, out, e2);
    });
}

}  // namespace strata::ds4::net
