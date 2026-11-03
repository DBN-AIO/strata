#!/usr/bin/env python3
"""Sparse slice of a GGUF shard for an expert server: header + every non-expert tensor + the routed-expert tensors of
layers lo..hi.  Everything else is a hole (reads as zeros) at the same offsets, so the engine opens the file as the
whole model.  Usage: slice_gguf.py SRC DST LO HI   (run from the Strata checkout: uses tools/gguf_reader.py)"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import GGUFFile

src, dst, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
g = GGUFFile(src)
size = os.path.getsize(src)
ts = sorted(g.tensors, key=lambda t: t.offset)
spans = []
for i, t in enumerate(ts):
    end = ts[i + 1].offset if i + 1 < len(ts) else size - g.data_start
    spans.append((t, g.data_start + t.offset, end - t.offset))
exp = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
keep = [(0, g.data_start)]
kept_exp = 0
for t, off, n in spans:
    m = exp.match(t.name)
    if m:
        if lo <= int(m.group(1)) <= hi:
            keep.append((off, n)); kept_exp += n
    else:
        keep.append((off, n))
total = sum(n for _, n in keep)
with open(src, "rb") as f, open(dst, "wb") as o:
    o.truncate(size)
    for off, n in sorted(keep):
        f.seek(off); o.seek(off)
        left = n
        while left > 0:
            b = f.read(min(left, 64 << 20))
            if not b:
                break
            o.write(b); left -= len(b)
print("sliced %s -> %s: layers %d-%d experts %.2f GiB, real bytes %.2f GiB of %.2f" %
      (os.path.basename(src), dst, lo, hi, kept_exp / 2**30, total / 2**30, size / 2**30))
