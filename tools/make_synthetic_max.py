#!/usr/bin/env python3
"""A Qwen3.8-Max-SHAPED synthetic GGUF: the real model's layer structure, expert count, top-k, tensor names, tensor types and
metadata keys, with a tiny hidden size and random weights.  It lets the engine's Max path (qwen35moe, 92 + 1 MTP blocks, 512
experts, top-10, IQ1_M / IQ2_XXS / Q2_K experts, Q5_K / Q6_K / Q8_0 / Q4_K dense weights) run end to end without the 400-700 GB
download: tests/qwen_max/run_synthetic.sh builds one and runs it single-process and through an expert server.

What is the real Max (read from the GGUF headers of unsloth/Qwen3.8-2.4T-A95B-GGUF, docs/QWEN_MAX.md) and what is scaled down:

    kept from Max                                 Max                     default here
    --layers (trunk blocks, multiple of 4)        92                      92
    MTP blocks (nextn_predict_layers)             1                       1
    full_attention_interval                       4                       4
    experts per layer / top-k                     512 / 10                512 / 10
    head_dim, ssm state, conv kernel, rope dim    256, 128, 4, 64         256, 128, 4, 64   (the kernels' compiled widths)
    vocabulary                                    248320                  248320
    scaled down (flags)
    --n-embd                                      8192                    256
    --n-ff (routed and shared expert width)       2048                    256
    --heads / --kv-heads                          64 / 4                  16 / 1            (the same 16:1 grouping)
    --ssm-k-heads / --ssm-v-heads                 16 / 128                2 / 16            (the same 1:8 ratio)

The weights are random BYTES of each quantization type with the scale fields set (fp16 d, dmin) so the dequantised values
have a chosen standard deviation (--gain / sqrt(fan_in)): not a model that says anything, but one whose activations stay
finite through 93 blocks, whose router spreads over all experts, and whose greedy output depends on every layer's experts.
Same seed, same file, byte for byte.

    python tools/make_synthetic_max.py OUT.gguf [--layers 92] [--quant iq1_m|iq2_xxs] [--seed 1] ...

A Max-shaped file with the real widths is NOT written by this tool (it would be the 400 GB model).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import BLOCK_GEOMETRY  # noqa: E402

TYPE_ID = {"F32": 0, "Q8_0": 8, "Q2_K": 10, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ2_XXS": 16, "IQ1_M": 29}
# the standard deviation of one dequantised weight for d = 1 when the scale fields are set as in `blocks()` below
STD = {"Q8_0": 74.0, "Q2_K": 13.9, "Q4_K": 258.0, "Q5_K": 527.0, "Q6_K": 85.7, "IQ2_XXS": 66.9, "IQ1_M": 7.1}
ALIGN = 32


def nbytes(typ: str, elems: int) -> int:
    if typ == "F32":
        return elems * 4
    be, bb = BLOCK_GEOMETRY[typ]
    if elems % be:
        raise ValueError(f"{typ}: {elems} elements is not a multiple of {be}")
    return elems // be * bb


def rng_for(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def f16_bytes(x: float) -> np.ndarray:
    return np.frombuffer(np.float16(x).tobytes(), dtype=np.uint8)


def blocks(typ: str, n_blocks: int, d: float, rng: np.random.Generator) -> np.ndarray:
    """n_blocks random blocks of `typ` as a (n_blocks, bytes) uint8 array with the scale fields set from `d`."""
    be, bb = BLOCK_GEOMETRY[typ]
    b = rng.integers(0, 256, size=(n_blocks, bb), dtype=np.uint8)
    if typ == "Q8_0":                       # d | 32 x int8
        b[:, 0:2] = f16_bytes(d)
    elif typ in ("Q4_K", "Q5_K"):           # d | dmin | 12 scale bytes | (qh) | qs.  dmin = d * mean(q) makes the weights zero-mean
        b[:, 0:2] = f16_bytes(d)
        b[:, 2:4] = f16_bytes(d * (7.5 if typ == "Q4_K" else 15.5))
    elif typ == "Q6_K":                     # ql[128] | qh[64] | int8 scales[16] | d
        b[:, 192:208] = rng.integers(-8, 8, size=(n_blocks, 16), dtype=np.int8).view(np.uint8)
        b[:, 208:210] = f16_bytes(d)
    elif typ == "Q2_K":                     # scales[16] | qs[64] | d | dmin
        b[:, 80:82] = f16_bytes(d)
        b[:, 82:84] = f16_bytes(d * 1.5)
    elif typ == "IQ2_XXS":                  # d | 8 x (uint32 grid indices, uint32 signs + 4-bit scale)
        b[:, 0:2] = f16_bytes(d)
    elif typ == "IQ1_M":                    # qs[32] | qh[16] | 4 x uint16 scales whose top nibbles carry the fp16 d
        bits = int(np.frombuffer(np.float16(d).tobytes(), dtype=np.uint16)[0])
        for i in range(4):
            b[:, 49 + 2 * i] = (b[:, 49 + 2 * i] & 0x0F) | np.uint8(((bits >> (4 * i)) & 0xF) << 4)
    else:
        raise ValueError(typ)
    return b


class Tensor:
    def __init__(self, name, shape, typ, gain=1.0, kind="rand"):
        self.name, self.shape, self.typ, self.gain, self.kind = name, list(shape), typ, gain, kind
        self.elems = int(np.prod(shape))
        self.nbytes = nbytes(typ, self.elems)

    def data(self, seed: int) -> bytes:
        rng = rng_for(seed, self.name)
        if self.typ == "F32":
            return self.f32(rng).astype("<f4").tobytes()
        be, _ = BLOCK_GEOMETRY[self.typ]
        fan_in = self.shape[0]
        d = self.gain / np.sqrt(fan_in) / STD[self.typ]
        out = bytearray()
        step = 1 << 20   # blocks per chunk
        n = self.elems // be
        for s in range(0, n, step):
            out += blocks(self.typ, min(step, n - s), d, rng).tobytes()
        assert len(out) == self.nbytes
        return bytes(out)

    def f32(self, rng) -> np.ndarray:
        n = self.elems
        if self.kind == "ones":
            return 1.0 + 0.05 * rng.standard_normal(n)
        if self.kind == "ssm_a":            # -exp(A_log), A in [1, 16]
            return -np.exp(rng.uniform(0.0, np.log(16.0), n))
        if self.kind == "dt_bias":
            return rng.uniform(-3.0, -0.5, n)
        return rng.standard_normal(n) * (self.gain / np.sqrt(self.shape[0]))


def build_tensors(a) -> list[Tensor]:
    N, FF, NE = a.n_embd, a.n_ff, a.experts
    H, KV, HD = a.heads, a.kv_heads, 256
    SK, SV, SS = a.ssm_k_heads, a.ssm_v_heads, 128
    VAL = SV * SS
    CONV = 2 * SS * SK + VAL
    q = a.quant.upper()
    ex_trunk = {"IQ1_M": "IQ1_M", "IQ2_XXS": "IQ2_XXS"}[q]
    T = [Tensor("token_embd.weight", [N, a.vocab], "Q4_K", gain=1.0),
         Tensor("output_norm.weight", [N], "F32", kind="ones"),
         Tensor("output.weight", [N, a.vocab], "Q4_K", gain=3.0)]
    total = a.layers + a.mtp
    for l in range(total):
        p = f"blk.{l}."
        mtp = l >= a.layers
        attn = mtp or (l % a.interval == a.interval - 1)
        ex = "Q2_K" if mtp else ex_trunk
        T += [Tensor(p + "attn_norm.weight", [N], "F32", kind="ones"),
              Tensor(p + "post_attention_norm.weight", [N], "F32", kind="ones"),
              Tensor(p + "ffn_gate_inp.weight", [N, NE], "F32", gain=2.0),
              Tensor(p + "ffn_gate_inp_shexp.weight", [N], "F32", gain=1.0),
              Tensor(p + "ffn_gate_shexp.weight", [N, FF], "Q5_K", gain=1.0),
              Tensor(p + "ffn_up_shexp.weight", [N, FF], "Q5_K", gain=1.0),
              Tensor(p + "ffn_down_shexp.weight", [FF, N], "Q6_K", gain=0.5),
              Tensor(p + "ffn_gate_exps.weight", [N, FF, NE], ex, gain=1.0),
              Tensor(p + "ffn_up_exps.weight", [N, FF, NE], ex, gain=1.0),
              Tensor(p + "ffn_down_exps.weight", [FF, N, NE], ex, gain=0.5)]
        if attn:
            T += [Tensor(p + "attn_q.weight", [N, 2 * H * HD], "Q5_K", gain=1.0),
                  Tensor(p + "attn_k.weight", [N, KV * HD], "Q5_K", gain=1.0),
                  Tensor(p + "attn_v.weight", [N, KV * HD], "Q5_K", gain=1.0),
                  Tensor(p + "attn_output.weight", [H * HD, N], "Q5_K", gain=0.5),
                  Tensor(p + "attn_q_norm.weight", [HD], "F32", kind="ones"),
                  Tensor(p + "attn_k_norm.weight", [HD], "F32", kind="ones")]
        else:
            T += [Tensor(p + "attn_qkv.weight", [N, CONV], "Q5_K", gain=1.0),
                  Tensor(p + "attn_gate.weight", [N, VAL], "Q5_K", gain=1.0),
                  Tensor(p + "ssm_out.weight", [VAL, N], "Q6_K", gain=0.5),
                  Tensor(p + "ssm_conv1d.weight", [4, CONV], "F32", gain=2.0),
                  Tensor(p + "ssm_alpha.weight", [N, SV], "F32", gain=1.0),
                  Tensor(p + "ssm_beta.weight", [N, SV], "F32", gain=1.0),
                  Tensor(p + "ssm_a", [SV], "F32", kind="ssm_a"),
                  Tensor(p + "ssm_dt.bias", [SV], "F32", kind="dt_bias"),
                  Tensor(p + "ssm_norm.weight", [SS], "F32", kind="ones")]
        if mtp:
            T += [Tensor(p + "nextn.eh_proj.weight", [2 * N, N], "Q8_0", gain=0.7),
                  Tensor(p + "nextn.enorm.weight", [N], "F32", kind="ones"),
                  Tensor(p + "nextn.hnorm.weight", [N], "F32", kind="ones"),
                  Tensor(p + "nextn.shared_head_norm.weight", [N], "F32", kind="ones")]
    return T


# ------------------------------------------------------------------ GGUF v3 writer (streaming)
def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, typ, val) -> bytes:
    out = s(key)
    if typ == "str":
        return out + struct.pack("<I", 8) + s(val)
    if typ == "u32":
        return out + struct.pack("<II", 4, val)
    if typ == "f32":
        return out + struct.pack("<If", 6, val)
    if typ == "arr_str":
        return out + struct.pack("<IIQ", 9, 8, len(val)) + b"".join(s(v) for v in val)
    if typ == "arr_i32":
        return out + struct.pack("<IIQ", 9, 5, len(val)) + struct.pack(f"<{len(val)}i", *val)
    raise ValueError(typ)


def metadata(a) -> list[bytes]:
    p = "qwen35moe."
    ssm_val = a.ssm_v_heads * 128
    m = [kv("general.architecture", "str", "qwen35moe"), kv("general.type", "str", "model"),
         kv("general.name", "str", "Qwen3.8-2.4T-A95B-synthetic"), kv("general.size_label", "str", "synthetic-max-shaped"),
         kv("general.quantization_version", "u32", 2),
         kv(p + "block_count", "u32", a.layers + a.mtp), kv(p + "context_length", "u32", 262144),
         kv(p + "embedding_length", "u32", a.n_embd), kv(p + "attention.head_count", "u32", a.heads),
         kv(p + "attention.head_count_kv", "u32", a.kv_heads), kv(p + "rope.dimension_sections", "arr_i32", [11, 11, 10, 0]),
         kv(p + "rope.freq_base", "f32", 10000000.0), kv(p + "attention.layer_norm_rms_epsilon", "f32", 1e-6),
         kv(p + "expert_count", "u32", a.experts), kv(p + "expert_used_count", "u32", a.top_k),
         kv(p + "attention.key_length", "u32", 256), kv(p + "attention.value_length", "u32", 256),
         kv(p + "expert_feed_forward_length", "u32", a.n_ff), kv(p + "expert_shared_feed_forward_length", "u32", a.n_ff),
         kv(p + "nextn_predict_layers", "u32", a.mtp), kv(p + "ssm.conv_kernel", "u32", 4), kv(p + "ssm.state_size", "u32", 128),
         kv(p + "ssm.group_count", "u32", a.ssm_k_heads), kv(p + "ssm.time_step_rank", "u32", a.ssm_v_heads),
         kv(p + "ssm.inner_size", "u32", ssm_val), kv(p + "full_attention_interval", "u32", a.interval),
         kv(p + "rope.dimension_count", "u32", 64),
         kv("tokenizer.ggml.model", "str", "gpt2"), kv("tokenizer.ggml.pre", "str", "qwen35"),
         kv("tokenizer.ggml.tokens", "arr_str", [f"<t{i}>" for i in range(a.vocab)])]
    return m


def write(a) -> None:
    T = build_tensors(a)
    kvs = metadata(a)
    off, infos = 0, []
    for t in T:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        info = s(t.name) + struct.pack("<I", len(t.shape)) + struct.pack(f"<{len(t.shape)}Q", *t.shape) \
            + struct.pack("<IQ", TYPE_ID[t.typ], off)
        infos.append(info)
        t.offset = off
        off += t.nbytes
    head = b"GGUF" + struct.pack("<IQQ", 3, len(T), len(kvs)) + b"".join(kvs) + b"".join(infos)
    pad = (-len(head)) % ALIGN
    with open(a.out, "wb") as f:
        f.write(head + b"\0" * pad)
        pos = 0
        for t in T:
            if t.offset > pos:
                f.write(b"\0" * (t.offset - pos))
            f.write(t.data(a.seed))
            pos = t.offset + t.nbytes
    n_exp = sum(t.nbytes for t in T if "_exps." in t.name)
    print(f"{a.out}: {len(T)} tensors, {os.path.getsize(a.out) / 2**30:.2f} GiB (experts {n_exp / 2**30:.2f} GiB), "
          f"{a.layers}+{a.mtp} blocks, {a.experts} experts top-{a.top_k}, n_embd {a.n_embd}, ff {a.n_ff}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=92, help="trunk blocks (Max: 92)")
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-ff", type=int, default=256)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--ssm-k-heads", type=int, default=2)
    ap.add_argument("--ssm-v-heads", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--quant", choices=("iq1_m", "iq2_xxs"), default="iq1_m", help="the routed experts' type (the MTP block's are Q2_K, as in Max)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.layers % a.interval:
        ap.error("--layers must be a multiple of the full-attention interval")
    for v, w in ((a.n_embd, "--n-embd"), (a.n_ff, "--n-ff")):
        if v % 256:
            ap.error(f"{w} must be a multiple of 256 (the K / IQ block width)")
    if (a.ssm_v_heads * 128) % 256:
        ap.error("ssm_v_heads * 128 must be a multiple of 256")
    write(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
