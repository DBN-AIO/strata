// src/core/native_rows.cpp - see include/strata/core/native_rows.hpp.
#include "strata/core/native_rows.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/core/net_proto.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>

namespace strata::core {

namespace np = netproto;
namespace C = strata::kernels::cpu;

NetIdentity native_net_identity(const strata::GgufModel& gm, const ModelGeometry& g, int64_t top_k, const std::vector<std::string>& files) {
    np::Sha256 h;
    auto u64 = [&](uint64_t v) { h.update(&v, 8); };
    h.update("SXPFPQ1", 7);   // the qwen35moe family's expert layout (Flash-Next's is "SXPFP2")
    u64((uint64_t) g.n_layers); u64((uint64_t) g.n_mtp_layers); u64((uint64_t) g.n_embd); u64((uint64_t) g.n_ff);
    u64((uint64_t) g.n_expert); u64((uint64_t) top_k);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const char* part : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const strata::TensorInfo* t = gm.find("blk." + std::to_string(l) + "." + part);
            if (t == nullptr) { u64(~0ull); continue; }
            u64(t->type); u64(t->shape.size());
            for (uint64_t d : t->shape) u64(d);
        }
    std::vector<std::string> seen;   // the files' sizes (not their paths: the hosts keep them elsewhere)
    for (const std::string& f : files) {
        if (std::find(seen.begin(), seen.end(), f) != seen.end()) continue;
        seen.push_back(f);
        std::error_code ec;
        const uint64_t sz = (uint64_t) std::filesystem::file_size(f, ec);
        u64(ec ? ~0ull : sz);
    }
    u64(seen.size());
    NetIdentity id;
    h.finish(id.fingerprint);
    id.isa = net_isa_bits();
    return id;
}

bool native_expert_blob(const strata::GgufModel& gm, int64_t layer, int64_t e, const C::NativeFmt& fmt, uint8_t* dst, std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    size_t sg = 0, su = 0, sd = 0;
    const strata::TensorInfo* tg = gm.find(p + "ffn_gate_exps.weight", &sg);
    const strata::TensorInfo* tu = gm.find(p + "ffn_up_exps.weight", &su);
    const strata::TensorInfo* td = gm.find(p + "ffn_down_exps.weight", &sd);
    if (!tg || !tu || !td) { err = "missing the expert tensors of block " + std::to_string(layer); return false; }
    if ((int) tg->type != fmt.gu_type || (int) tu->type != fmt.gu_type || (int) td->type != fmt.d_type) {
        err = "block " + std::to_string(layer) + ": expert types differ from the layout";
        return false;
    }
    const size_t gsz = (size_t) fmt.n_ff * fmt.gu_row, dsz = (size_t) fmt.n_embd * fmt.d_row;
    std::memcpy(dst, gm.shard(sg).tensor_data(*tg) + (size_t) e * gsz, gsz);
    std::memcpy(dst + fmt.up_off, gm.shard(su).tensor_data(*tu) + (size_t) e * gsz, gsz);
    std::memcpy(dst + fmt.down_off, gm.shard(sd).tensor_data(*td) + (size_t) e * dsz, dsz);
    return true;
}

bool NativeRows::run(int layer, int n_tok, int k, const int32_t* ids, const uint8_t* mask, const float* x, float* rows,
                     C::NativeArithmetic arith, std::string& err) {
    if (layer < 0 || (size_t) layer >= layers_.size() || layers_[(size_t) layer].base == nullptr) {
        err = "block " + std::to_string(layer) + " is not held here";
        return false;
    }
    const NativeLayerBlobs& L = layers_[(size_t) layer];
    const C::NativeFmt& f = L.fmt;
    const size_t N = (size_t) n_embd_;
    constexpr size_t AB = C::kNativeActBytes;
    acts_.resize((size_t) n_tok * AB + 64);
    uint8_t* acts = acts_.data() + ((64 - (reinterpret_cast<uintptr_t>(acts_.data()) & 63)) & 63);
    std::vector<char> quantized((size_t) n_tok, 0);
    std::vector<C::ExpertJobMulti> jobs;
    std::map<int, size_t> open;   // expert -> its job with room (up to MAXT tokens per job)
    for (int64_t i = 0; i < (int64_t) n_tok * k; ++i) {
        if (!mask[i]) continue;
        const int ex = ids[i];
        if (ex < 0 || ex >= n_expert_) { err = "expert id outside the model"; return false; }
        const int t = (int) (i / k);
        if (!quantized[(size_t) t]) {
            C::native_quant_act(f, x + (size_t) t * N, acts + (size_t) t * AB);
            quantized[(size_t) t] = 1;
        }
        auto it = open.find(ex);
        if (it == open.end() || jobs[it->second].nt == C::MAXT) {
            C::ExpertJobMulti j;
            j.blob = L.base + (size_t) ex * L.stride;
            jobs.push_back(j);
            it = open.insert_or_assign(ex, jobs.size() - 1).first;
        }
        C::ExpertJobMulti& j = jobs[it->second];
        j.nact[j.nt] = acts + (size_t) t * AB;
        j.out[j.nt] = rows + (size_t) i * N;
        ++j.nt;
    }
    if (jobs.empty()) return true;
    pool_.run_split_multi_native(f, jobs.data(), (int) jobs.size(), arith);
    return true;
}

}  // namespace strata::core
