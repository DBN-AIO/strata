#!/usr/bin/env python3
"""A synthetic DeepSeek-V4.1-Flash-shaped expert pack for the network split test (docs/DEEPSEEK41_SPLIT.md).

Same structure as a real pack's routed experts: layers, 'n_routed_experts' experts a layer, top-k picks, gate (w1) / up (w3) /
down (w2) matrices in the quant types the real sets use (IQ2_XXS, IQ3_XXS, IQ4_XS, Q4_K, MXFP4), each matrix of every expert at
a fixed stride from expert 0's in one file, a tensors.txt and ds4.txt that src/dsv41/ds4_net.cpp (and the engine's pack
reader, for the expert part) read.  Tiny hidden size (256), random weights with finite scales.  The dense part (router, shared
expert) lives in tests/dsv41/ds4_net_test.cpp.

  python tools/dsv41/make_synthetic_pack.py OUT_DIR [--layers 8] [--experts 32] [--topk 6] [--seed 1]
"""
import argparse
import os
import random
import struct

# name -> (ggml id, values a block, bytes a block, offsets of the fp16 scales to make finite; None: MXFP4's e8m0 byte)
TYPES = {
    "IQ1_M": (29, 256, 56, "iq1m"),        # no d field: the fp16 scale is spread over the top nibbles of 4 scale words (bytes 48..55)
    "IQ2_XXS": (16, 256, 66, [0]),
    "IQ2_XS": (17, 256, 74, [0]),
    "IQ3_XXS": (18, 256, 98, [0]),
    "IQ4_XS": (23, 256, 136, [0]),
    "Q4_K": (12, 256, 144, [0, 2]),
    "MXFP4": (39, 32, 17, None),
}
# (gate/up, down) of each layer, cycled
# (the real 2.00 bpw set mixes IQ1_M / IQ2_XXS / IQ2_XS this way: gate = up, down another type; a 3 bit set is IQ3_XXS)
CYCLE = [("IQ1_M", "IQ1_M"), ("IQ2_XXS", "IQ1_M"), ("IQ2_XS", "IQ1_M"), ("IQ2_XXS", "IQ2_XS"),
         ("IQ3_XXS", "IQ3_XXS"), ("IQ3_XXS", "IQ4_XS"), ("Q4_K", "Q4_K"), ("MXFP4", "MXFP4")]


def matrix(rng, ty, n, k, n_exp, fast=False):
    gid, vals, bb, scales = TYPES[ty]
    assert k % vals == 0
    row = (k // vals) * bb
    total = n_exp * n * row
    buf = bytearray(rng.randbytes(total))
    nb = total // bb
    if fast:   # one finite scale for every block (timing runs at real sizes: no per-block loop)
        d = struct.pack("<e", 0.006)
        if scales is None:
            buf[0::bb] = bytes([120]) * nb
        elif scales == "iq1m":
            h = struct.unpack("<H", d)[0]
            for i in range(4):
                buf[49 + 2 * i::bb] = bytes([(buf[49 + 2 * i] & 0x0F) | (((h >> (4 * i)) & 0xF) << 4)]) * nb
        else:
            for so in scales:
                buf[so::bb] = bytes([d[0]]) * nb
                buf[so + 1::bb] = bytes([d[1]]) * nb
        return buf, n * row
    for blk in range(nb):
        o = blk * bb
        if scales is None:
            buf[o] = rng.randrange(116, 125)
        elif scales == "iq1m":
            h = struct.unpack("<H", struct.pack("<e", 0.002 + 0.01 * rng.random()))[0]
            for i in range(4):
                buf[o + 49 + 2 * i] = (buf[o + 49 + 2 * i] & 0x0F) | (((h >> (4 * i)) & 0xF) << 4)
        else:
            for so in scales:
                buf[o + so:o + so + 2] = struct.pack("<e", 0.002 + 0.01 * rng.random())
    return buf, n * row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--types", help="GATE_UP,DOWN: every layer of this type pair (default: the real 2.00 set's mix, cycled)")
    ap.add_argument("--fast", action="store_true", help="one scale for all blocks (for timing runs at the real 5120 x 2304 size)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model = "model.bin"
    lines = ["file 0 " + model]
    off = 0
    with open(os.path.join(a.out, model), "wb") as f:
        for il in range(a.layers):
            tgu, td = tuple(a.types.split(",")) if a.types else CYCLE[il % len(CYCLE)]
            for w, ty, n, k in (("w1", tgu, a.ff, a.dim), ("w3", tgu, a.ff, a.dim), ("w2", td, a.dim, a.ff)):
                buf, per = matrix(rng, ty, n, k, a.experts, a.fast)
                f.write(buf)
                for e in range(a.experts):
                    lines.append("t layers.%d.ffn.experts.%d.%s.weight %s 0 %d %d %d %d" % (il, e, w, ty, off + e * per, per, n, k))
                off += len(buf)
    with open(os.path.join(a.out, "tensors.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    kv = {
        "hidden_size": a.dim, "moe_intermediate_size": a.ff, "num_hidden_layers": a.layers, "n_routed_experts": a.experts,
        "num_experts_per_tok": a.topk, "swiglu_limit": "10.0", "format": "gguf",
        "model_dir": os.path.abspath(a.out).replace("\\", "/"),
    }
    with open(os.path.join(a.out, "ds4.txt"), "w") as f:
        for k, v in kv.items():
            f.write("%s %s\n" % (k, v))
    print("synthetic pack: %d layers x %d experts, top-%d, hidden %d, ffn %d, %.1f MiB in %s" % (a.layers, a.experts, a.topk, a.dim, a.ff, off / 2**20, a.out))


if __name__ == "__main__":
    main()
