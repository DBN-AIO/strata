#!/usr/bin/env python3
"""Where DeepSeek-V4.1-Flash's expert layers go across the fleet (docs/DEEPSEEK41_SPLIT.md).

Reads the size of every layer's routed experts (from a pack's tensors.txt, or --uniform-gb for a set that is not on disk) and the
hosts' budgets (RAM + VRAM usable for experts, free disk), gives each host a CONTIGUOUS range of layers in the order conductor,
rig, T340, stream PC (the order of the conductor's own hops), and prints the `--remote-experts` line.  A host takes layers while
both its memory and its disk budget last.  When the layers do not fit it says by how many GB and exits 3, so a launcher refuses the
run instead of starting a model that pages from the SSD at every token.

  plan_split.py PACK                          # the fleet as inventoried 2026-11-03
  plan_split.py PACK --drop stream            # stream mode: the stream PC's server is stopped, the others take its layers (or: shortfall)
  plan_split.py --uniform-gb 5.2022           # a 3 bit set (IQ3_XXS everywhere): 40 layers of 5.2 GB
  plan_split.py PACK --hosts fleet.json       # [{"name","addr","mem_gb","disk_gb","compute_ms","wire_ms"}, ...] in hop order
"""
import argparse
import glob
import json
import struct
import sys

# usable for experts, GB.  mem = RAM the OS can keep as page cache + VRAM the server/conductor may fill; disk = free space for the slice.
# compute_ms: one layer-call (6 experts, one token, 2.00 bpw mix, resident, CPU kernels, all threads) measured with ds4_net_test on a
# synthetic pack of the real 5120 x 2304 shape; wire_ms: 21.6 KB out + 129 KB back on the house LAN (0 for the conductor itself).
FLEET = [
    {"name": "gaming", "addr": None, "mem_gb": 87.0, "disk_gb": 1e9, "compute_ms": 2.5, "wire_ms": 0.0,
     "note": "conductor: 35 GB of VRAM for hot experts (48 - dense - KV) + 52 GB page cache; its layers are its own CPU lane / GPU"},
    {"name": "rig", "addr": "192.168.1.130:7840", "mem_gb": 13.0, "disk_gb": 20.0, "compute_ms": 4.5, "wire_ms": 1.2,
     "note": "32 GB RAM with the fleet's TabbyAPI cards loaded (15 GB free); 24 GB free disk"},
    {"name": "t340", "addr": "192.168.1.133:7840", "mem_gb": 30.0, "disk_gb": 14.0, "compute_ms": 6.4, "wire_ms": 0.75,
     "note": "64 GB RAM, AIO services use 22; 17 GB free disk (the binding limit)"},
    {"name": "stream", "addr": "192.168.1.27:7840", "mem_gb": 48.0, "disk_gb": 24.0, "compute_ms": 2.9, "wire_ms": 0.9,
     "note": "64 GB RAM, ~16 GB used by the stream; C: has 27 GB free (D: is failing); no VRAM until TabbyAPI's 11 GB are freed"},
]


# ggml: values and bytes of one block
BLOCK = {10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98),
         19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 39: (32, 17)}


def layer_gb_from_gguf(dirpath):
    """Expert bytes per layer from the GGUF shards' headers only (tensor infos): blk.N.ffn_{gate,up,down}_exps.weight."""
    per = {}
    for fn in sorted(glob.glob(dirpath.rstrip("/\\") + "/*.gguf")):
        f = open(fn, "rb")
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        if f.read(4) != b"GGUF":
            continue
        rd("I")
        nt, nkv = rd("QQ")
        rstr = lambda: f.read(rd("Q")[0]).decode("utf-8", "replace")
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t in sizes:
                f.seek(sizes[t], 1)
            elif t == 8:
                rstr()
            elif t == 9:
                et, n = rd("IQ")
                for _ in range(n):
                    skip(et)
        for _ in range(nkv):
            rstr()
            skip(rd("I")[0])
        for _ in range(nt):
            name = rstr()
            nd = rd("I")[0]
            dims = rd("Q" * nd)
            ty = rd("I")[0]
            rd("Q")
            if "_exps.weight" in name and name.startswith("blk.") and ty in BLOCK:
                n = 1
                for d in dims:
                    n *= d
                v, b = BLOCK[ty]
                il = int(name.split(".")[1])
                per[il] = per.get(il, 0) + n // v * b
    if not per:
        sys.exit("no expert tensors in %s" % dirpath)
    return [per.get(i, 0) / 1e9 for i in range(max(per) + 1)]


def layer_gb_from_pack(pack):
    per = {}
    for line in open(pack.rstrip("/\\") + "/tensors.txt", encoding="utf-8"):
        if line.startswith("t layers.") and ".ffn.experts." in line:
            p = line.split()
            il = int(p[1].split(".")[1])
            per[il] = per.get(il, 0) + int(p[5])
    n = max(per) + 1
    return [per.get(i, 0) / 1e9 for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", nargs="?")
    ap.add_argument("--gguf-dir", help="a folder of GGUF shards (headers are read, nothing else): the layer sizes of a set not packed yet")
    ap.add_argument("--uniform-gb", type=float, help="every layer this many GB (no pack)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--hosts", help="JSON file replacing the built-in fleet")
    ap.add_argument("--drop", action="append", default=[], help="a host that is not serving (stream mode: stream)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.gguf_dir:
        gb = layer_gb_from_gguf(a.gguf_dir)
    elif a.pack:
        gb = layer_gb_from_pack(a.pack)
    elif a.uniform_gb:
        gb = [a.uniform_gb] * a.layers
    else:
        sys.exit(__doc__)
    hosts = json.load(open(a.hosts)) if a.hosts else FLEET
    hosts = [h for h in hosts if h["name"] not in a.drop]
    out, i = [], 0
    for h in hosts:
        cap = min(h["mem_gb"], h["disk_gb"])
        start, used = i, 0.0
        while i < len(gb) and used + gb[i] <= cap + 1e-9:
            used += gb[i]
            i += 1
        out.append({"host": h["name"], "addr": h["addr"], "first": start, "last": i - 1, "layers": i - start, "gb": round(used, 2),
                    "limit": "disk" if h["disk_gb"] < h["mem_gb"] else "memory", "cap_gb": cap})
    short = sum(gb[i:])
    ms = 0.0
    for r, h in zip(out, hosts):
        ms += r["layers"] * (h["compute_ms"] + h["wire_ms"])
    spec = ",".join("%s=%d-%d" % (r["addr"], r["first"], r["last"]) for r in out if r["addr"] and r["layers"] > 0)
    res = {"total_gb": round(sum(gb), 2), "plan": out, "remote_experts": spec, "shortfall_gb": round(short, 2), "unplaced_layers": len(gb) - i,
           "expert_ms_per_token": round(ms, 1), "note": "CPU kernels everywhere, layer after layer; add the conductor's dense time per token"}
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print("experts: %.1f GB in %d layers" % (sum(gb), len(gb)))
        for r in out:
            print("  %-7s layers %s  %5.1f GB of %5.1f (%s-limited)" % (r["host"], ("%d-%d" % (r["first"], r["last"])) if r["layers"] else "-", r["gb"], r["cap_gb"], r["limit"]))
        print("  --remote-experts %s" % (spec or "(none)"))
        print("  experts alone, one token: %.1f ms (%.1f tok/s ceiling)" % (ms, 1000.0 / ms if ms else 0))
    if short > 1e-9:
        sys.stderr.write("REFUSED: %d layers (%.1f GB) have no host: free disk or memory on %s, or serve them from the conductor's SSD\n"
                         % (len(gb) - i, short, ", ".join(h["name"] for h in hosts)))
        sys.exit(3)


if __name__ == "__main__":
    main()
