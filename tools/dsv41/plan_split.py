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
  plan_split.py PACK --hosts fleet.json       # [{"name","addr","mem_gb","disk_gb","compute_ms","wire_ms"[,"vram_gb","gpu_ms"]}, ...] in hop order

BBXL S3, popularity-aware placement (docs/DEEPSEEK41_SPLIT.md "Popularity-aware placement"):

  plan_split.py PACK --popularity expert-profile.bin [--perm-out perm.bin] [--groups N] [--allow-ssd]

--popularity is the engine's own expert profile: raw uint32 hit counts [layers (+ up to 8 drafter rows)][experts], written by
`strata-dsv41 --expert-profile FILE --profile-run IDS` (a FILE that does not exist starts at zero, so the file holds only that
run's prompts), or a TSV `layer<TAB>expert<TAB>count`.  The hottest (layer, expert) pairs go to the conductor's VRAM, then its RAM,
then the remote hosts fastest first (GPU servers, then CPU servers by call time), the coldest to the conductor's SSD (only with
--allow-ssd; otherwise the run is refused when the hosts' memory and disk do not hold everything).  The wire serves contiguous
expert ranges, so each layer's experts are RENUMBERED hottest first (--perm-out: the map the conductor translates its remote
calls with and `slice_stream.py emit --perm` lays the slice out by) and a host serves ranks e0-e1 of its layers.  A flat file
(every layer's counts equal) gives today's layer-major plan.
"""
import argparse
import glob
import json
import math
import os
import struct
import sys

# usable for experts, GB.  mem = RAM the OS can keep as page cache + VRAM the server/conductor may fill; disk = free space for the slice.
# compute_ms: one layer-call (6 experts, one token, 2.00 bpw mix, resident, CPU kernels, all threads) measured with ds4_net_test on a
# synthetic pack of the real 5120 x 2304 shape; wire_ms: 21.6 KB out + 129 KB back on the house LAN (0 for the conductor itself).
# vram_gb (optional, part of mem_gb): experts held in VRAM; gpu_ms: the same layer-call on the GPU (0.416 ms measured on the rig's 3090,
# --expert-server-gpu, docs/DEEPSEEK41_SPLIT.md "GPU expert server"); ssd_mbps: the conductor's cold-read speed (NVMe).
FLEET = [
    {"name": "gaming", "addr": None, "mem_gb": 87.0, "vram_gb": 35.0, "disk_gb": 1e9, "compute_ms": 2.5, "wire_ms": 0.0,
     "note": "conductor: 35 GB of VRAM for hot experts (48 - dense - KV) + 52 GB page cache; its layers are its own CPU lane / GPU"},
    {"name": "rig", "addr": "192.168.1.130:7840", "mem_gb": 13.0, "disk_gb": 20.0, "compute_ms": 4.5, "wire_ms": 1.2,
     "note": "32 GB RAM with the fleet's TabbyAPI cards loaded (15 GB free); 24 GB free disk"},
    {"name": "t340", "addr": "192.168.1.133:7840", "mem_gb": 30.0, "disk_gb": 14.0, "compute_ms": 6.4, "wire_ms": 0.75,
     "note": "64 GB RAM, AIO services use 22; 17 GB free disk (the binding limit)"},
    {"name": "stream", "addr": "192.168.1.27:7840", "mem_gb": 48.0, "disk_gb": 24.0, "compute_ms": 2.9, "wire_ms": 0.9,
     "note": "64 GB RAM, ~16 GB used by the stream; C: has 27 GB free (D: is failing); no VRAM until TabbyAPI's 11 GB are freed"},
]
GPU_MS = 0.416       # one layer call (6 experts, 1 token) on a 3090, measured (S2)
SSD_MBPS = 3000.0    # NVMe sequential read, the cold-expert read speed of the conductor


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


def pack_geometry(pack):
    """(n_routed_experts, num_experts_per_tok) from the pack's ds4.txt; None where it does not say."""
    kv = {}
    try:
        for line in open(pack.rstrip("/\\") + "/ds4.txt", encoding="utf-8"):
            p = line.split(None, 1)
            if len(p) == 2:
                kv[p[0]] = p[1].strip()
    except OSError:
        pass
    g = lambda k: int(kv[k]) if k in kv and kv[k].isdigit() else None
    return g("n_routed_experts"), g("num_experts_per_tok")


# ---------------------------------------------------------------------------------------------------- the popularity file

class PopError(Exception):
    pass


def load_popularity(path, n_layers, n_exp):
    """-> counts[layer][expert] (ints) from the engine's profile (uint32 [layers + 0..8 drafter rows][experts]) or a TSV."""
    try:
        blob = open(path, "rb").read()
    except OSError as e:
        raise PopError("popularity file %s: %s" % (path, e.strerror or e))
    if blob[:4] == b"STRP":
        raise PopError("popularity file %s is a ranked profile of the Qwen engine (STRP), not hit counts: a ranking is not a frequency; "
                       "count with `strata-dsv41 --expert-profile FILE --profile-run IDS`" % path)
    rows = [[0] * n_exp for _ in range(n_layers)]
    text = None
    if path.lower().endswith((".tsv", ".txt")):
        try:
            text = blob.decode("ascii")
        except UnicodeDecodeError:
            raise PopError("popularity file %s: a TSV must be ASCII" % path)
    if text is not None:
        for ln, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split("\t")
            try:
                il, e, c = int(p[0]), int(p[1]), int(p[2])
                if len(p) != 3:
                    raise ValueError
            except (ValueError, IndexError):
                raise PopError("popularity file %s line %d: need layer<TAB>expert<TAB>count" % (path, ln))
            if not (0 <= il < n_layers and 0 <= e < n_exp):
                raise PopError("popularity file %s line %d: (layer %d, expert %d) is outside %d layers x %d experts" % (path, ln, il, e, n_layers, n_exp))
            if c < 0:
                raise PopError("popularity file %s line %d: a negative count" % (path, ln))
            rows[il][e] += c
    else:
        row = 4 * n_exp
        if len(blob) < row * n_layers or len(blob) > row * (n_layers + 8) or len(blob) % 4:
            raise PopError("popularity file %s is %d bytes: %d layers x %d experts x 4 bytes = %d (a few drafter rows may follow, at most 8 rows); "
                           "another model's profile or a cut-off file" % (path, len(blob), n_layers, n_exp, row * n_layers))
        for il in range(n_layers):
            rows[il] = list(struct.unpack_from("<%dI" % n_exp, blob, il * row))
    if sum(sum(r) for r in rows) <= 0:
        raise PopError("popularity file %s has no hits at all" % path)
    return rows


def write_perm(path, perm, n_exp):
    """DS4PERM1, u32 layers, u32 experts, then uint16 rank[layer][original expert] (the conductor's wire id of an expert)."""
    with open(path, "wb") as f:
        f.write(b"DS4PERM1" + struct.pack("<II", len(perm), n_exp))
        for row in perm:
            f.write(struct.pack("<%dH" % n_exp, *row))


# ---------------------------------------------------------------------------------------------------- the model of one token

def _host_vram_ranks(hostpairs, vram_gb, sz):
    """Which of (layer, rank, p) pairs sit in VRAM: the hottest first until vram_gb; -> set of (layer, rank)."""
    out, used = set(), 0.0
    for l, r, p in sorted(hostpairs, key=lambda t: (-t[2], t[0], t[1])):
        if used + sz[l] > vram_gb + 1e-12:
            break
        used += sz[l]
        out.add((l, r))
    return out


def estimate(hosts, nl, ne, k, psorted, sz, assign, ssd_ms):
    """Expected ms of the experts of one token, from per-call measurements: layer l costs the slower of the conductor's own work and
    its slowest remote host (the calls are in flight together; the conductor computes its share while it waits).
    assign[l] = [(host index or -1 for the conductor's SSD, number of ranks), ...] in rank order, covering all ne ranks.
    psorted[l][r] = expected picks of the rank-r expert of layer l per token (sums to k a layer).  Returns (ms, hit share a host)."""
    per_host = {}
    for l in range(nl):
        r = 0
        for h, n in assign[l]:
            per_host.setdefault(h, []).extend((l, x, psorted[l][x]) for x in range(r, r + n))
            r += n
    vram = {h: _host_vram_ranks(prs, hosts[h].get("vram_gb", 0.0), sz) if h >= 0 else set() for h, prs in per_host.items()}
    total = 0.0
    share = {}
    for l in range(nl):
        local, remote, r = 0.0, 0.0, 0
        for h, n in assign[l]:
            seg = range(r, r + n)
            r += n
            ph = psorted[l]
            if h == -1:
                local += sum(ph[x] for x in seg) * ssd_ms[l]
            else:
                H = hosts[h]
                gms = H.get("gpu_ms", GPU_MS)
                cost = sum(ph[x] * ((gms if (l, x) in vram[h] else H["compute_ms"]) / k) for x in seg)
                if H["addr"] is None:
                    local += cost
                else:
                    miss = sum(math.log1p(-min(ph[x], 0.999999)) for x in seg)
                    remote = max(remote, (1.0 - math.exp(miss)) * H["wire_ms"] + cost)
            share[h] = share.get(h, 0.0) + sum(ph[x] for x in seg)
        total += max(local, remote)
    return total, {h: v / (k * nl) for h, v in share.items()}


def popularity_plan(a, hosts, gb, rows, n_exp, k, today_out):
    """The popularity-aware plan; -> dict for the report, or exits with the shortfall."""
    nl = len(gb)
    sz = [g / n_exp for g in gb]                      # GB of one expert of layer l
    # expected picks per token of each (layer, expert): each layer's counts scaled to k picks
    p = []
    for il in range(nl):
        s = sum(rows[il])
        p.append([k * c / s for c in rows[il]] if s > 0 else [k / n_exp] * n_exp)
    flat = all(max(r) == min(r) for r in rows)
    order = [sorted(range(n_exp), key=lambda e, il=il: (-rows[il][e], e)) for il in range(nl)]
    psorted = [[p[il][e] for e in order[il]] for il in range(nl)]
    perm = [[0] * n_exp for _ in range(nl)]
    for il in range(nl):
        for rank, e in enumerate(order[il]):
            perm[il][e] = rank
    cond = 0
    # cost of a cold expert read from NVMe: bytes / speed (GB / (MB/s / 1000) = s; x1000 = ms)
    ssd_ms = [sz[l] / (hosts[cond].get("ssd_mbps", SSD_MBPS) / 1e3) * 1e3 for l in range(nl)]

    def today_assign():
        asg = [None] * nl
        for r, h in zip(today_out, range(len(hosts))):
            for l in range(r["first"], r["last"] + 1):
                asg[l] = [(h, n_exp)]
        return asg

    placed_today = sum(r["layers"] for r in today_out) == nl
    base_ms = None
    if placed_today:
        base_ms, _ = estimate(hosts, nl, n_exp, k, psorted, sz, today_assign(), ssd_ms)

    # tiers in speed order; each (host index, capacity GB)
    remotes = [i for i in range(len(hosts)) if hosts[i]["addr"] is not None]
    remotes.sort(key=lambda i: (0 if hosts[i].get("vram_gb", 0.0) > 0 else 1, hosts[i]["compute_ms"] + hosts[i]["wire_ms"]))
    ccap = min(hosts[cond]["mem_gb"], hosts[cond]["disk_gb"])
    tiers = [(cond, ccap)] + [(i, min(hosts[i]["mem_gb"], hosts[i]["disk_gb"])) for i in remotes]
    tier_names = [hosts[i]["name"] for i, _ in tiers]
    total = sum(gb)
    short = total - sum(c for _, c in tiers)
    if short > 1e-9 and not a.allow_ssd:
        sys.stderr.write("REFUSED: %.1f GB of experts have no host: free disk or memory on %s, or pass --allow-ssd (the coldest experts are read "
                         "from the conductor's SSD)\n" % (short, ", ".join(h["name"] for h in hosts)))
        sys.exit(3)

    if flat:
        if not placed_today:
            sys.stderr.write("REFUSED: a flat popularity file gives today's layer-major plan, and it does not fit: free disk or memory on %s, or "
                             "give a real popularity file\n" %", ".join(h["name"] for h in hosts))
            sys.exit(3)
        asg_final = today_assign()
        remote_spec = None
    else:
        # fill: the hottest pairs into the first tier with room, in order
        pairs = sorted(((psorted[l][r], l, r) for l in range(nl) for r in range(n_exp)), key=lambda t: (-t[0], t[1], t[2]))
        room = [c for _, c in tiers]
        cnt = [[0] * (len(tiers) + 1) for _ in range(nl)]       # per layer: ranks in each tier, the last column = SSD
        ti = 0
        for pv, l, r in pairs:
            while ti < len(tiers) and room[ti] + 1e-12 < sz[l]:
                ti += 1
            if ti < len(tiers):
                room[ti] -= sz[l]
            cnt[l][ti] += 1
        # whole groups of layers share one expert range per host (a server serves ONE range of experts for all its layers)
        G = max(1, min(a.groups, nl))
        bounds = [round(i * nl / G) for i in range(G + 1)]
        gsz = [sum(sz[l] for l in range(bounds[g], bounds[g + 1])) for g in range(G)]
        tail = [0] * G
        seg = []                                                 # per group: ranks in each tier, the mean over its layers, rounded down
        for g in range(G):
            ls = range(bounds[g], bounds[g + 1])
            m = [int(sum(cnt[l][t] for l in ls) / len(ls) + 1e-9) for t in range(len(tiers))]
            if a.allow_ssd:
                tail[g] = int(sum(cnt[l][len(tiers)] for l in ls) / len(ls) + 1e-9)
            m[0] = n_exp - sum(m[1:]) - tail[g]                  # the conductor takes what rounding left over
            seg.append(m)
        # rounding down the remote tiers can leave the conductor over its budget: hand the conductor's coldest rank of a group to the
        # fastest remote tier that still has room, until it fits
        used = [sum(seg[g][t] * gsz[g] for g in range(G)) for t in range(len(tiers))]
        while used[0] > tiers[0][1] + 1e-9:
            best = None
            for t in range(1, len(tiers)):
                fit = [g for g in range(G) if seg[g][0] > 0 and used[t] + gsz[g] <= tiers[t][1] + 1e-9]
                if fit:
                    best = (t, min(fit, key=lambda g: gsz[g]))
                    break
            if best is None:
                break
            t, g = best
            seg[g][0] -= 1
            seg[g][t] += 1
            used[0] -= gsz[g]
            used[t] += gsz[g]
        assign = [None] * nl
        for g in range(G):
            for l in range(bounds[g], bounds[g + 1]):
                row = [(tiers[t][0], seg[g][t]) for t in range(len(tiers)) if seg[g][t] > 0]
                if tail[g]:
                    row.append((-1, tail[g]))
                assign[l] = row
        # the capacity check on what is really placed (rounding moved ranks)
        used = [0.0] * len(tiers)
        for l in range(nl):
            for (h, n) in assign[l]:
                if h == -1:
                    continue
                used[[t for t, (hh, _) in enumerate(tiers) if hh == h][0]] += n * sz[l]
        over = [(tier_names[t], used[t] - tiers[t][1]) for t in range(len(tiers)) if used[t] > tiers[t][1] + 1e-9]
        if over:
            sys.stderr.write("REFUSED: after rounding to %d expert range(s) per host the plan overflows %s; pass a larger --groups or free space\n"
                             % (G, ", ".join("%s by %.2f GB" % o for o in over)))
            sys.exit(3)
        asg_final = assign
        # routes: one item per (host, run of layers with the same rank range)
        items = []
        for g in range(G):
            l0 = bounds[g]
            e0 = 0
            for (h, n) in assign[l0]:
                if h == -1:
                    break
                if hosts[h]["addr"] is not None:
                    items.append((h, bounds[g], bounds[g + 1] - 1, e0, e0 + n - 1))
                e0 += n
        merged = []
        for it in items:
            if merged and merged[-1][0] == it[0] and merged[-1][3:] == it[3:] and merged[-1][2] + 1 == it[1]:
                merged[-1] = (it[0], merged[-1][1], it[2], it[3], it[4])
            else:
                merged.append(it)
        remote_spec = ",".join("%s=%d-%d/%d-%d" % (hosts[h]["addr"], lo, hi, e0, e1) for h, lo, hi, e0, e1 in merged)

    ms, share = estimate(hosts, nl, n_exp, k, psorted, sz, asg_final, ssd_ms)
    # per-host summary
    hrep = []
    for h, H in enumerate(hosts):
        segs, gbs, runs = [], 0.0, {}
        for l in range(nl):
            r = 0
            for (hh, n) in asg_final[l]:
                if hh == h and n > 0:
                    gbs += n * sz[l]
                    runs.setdefault((r, r + n - 1), []).append(l)
                r += n
        for (e0, e1), ls in sorted(runs.items(), key=lambda kv: kv[1][0]):
            lo = ls[0]
            for i in range(1, len(ls) + 1):
                if i == len(ls) or ls[i] != ls[i - 1] + 1:
                    segs.append({"layers": [lo, ls[i - 1]], "experts": [e0, e1]})
                    if i < len(ls):
                        lo = ls[i]
        hrep.append({"host": H["name"], "gb": round(gbs, 2), "hit_share": round(share.get(h, 0.0), 4), "segments": segs})
    ssd_share = round(share.get(-1, 0.0), 4)
    res = {"flat": flat, "experts": n_exp, "top_k": k, "groups": None if flat else max(1, min(a.groups, nl)), "hosts": hrep,
           "ssd_hit_share": ssd_share, "expected_ms_per_token": round(ms, 2),
           "baseline_ms_per_token": None if base_ms is None else round(base_ms, 2),
           "remote_experts": remote_spec, "perm": perm,
           "note": "expected ms = sum over layers of the slower of the conductor's own work and its slowest remote host, from the measured "
                   "per-call costs weighted by hit share; baseline = today's layer-major plan on the same hits"}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", nargs="?")
    ap.add_argument("--gguf-dir", help="a folder of GGUF shards (headers are read, nothing else): the layer sizes of a set not packed yet")
    ap.add_argument("--uniform-gb", type=float, help="every layer this many GB (no pack)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, help="routed experts a layer (default: the pack's n_routed_experts, else 384)")
    ap.add_argument("--top-k", type=int, help="experts picked a token (default: the pack's num_experts_per_tok, else 6)")
    ap.add_argument("--hosts", help="JSON file replacing the built-in fleet")
    ap.add_argument("--drop", action="append", default=[], help="a host that is not serving (stream mode: stream)")
    ap.add_argument("--popularity", help="expert hit counts (the engine's profile, or a TSV layer/expert/count): hottest experts to the fastest tier")
    ap.add_argument("--perm-out", help="with --popularity: write the expert renumbering (DS4PERM1) here")
    ap.add_argument("--groups", type=int, default=1, help="with --popularity: layer groups, each with its own expert ranges (one server per host and group)")
    ap.add_argument("--allow-ssd", action="store_true", help="with --popularity: the coldest experts may stay on the conductor's SSD")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pack_ne = pack_k = None
    if a.gguf_dir:
        gb = layer_gb_from_gguf(a.gguf_dir)
    elif a.pack:
        gb = layer_gb_from_pack(a.pack)
        pack_ne, pack_k = pack_geometry(a.pack)
    elif a.uniform_gb:
        gb = [a.uniform_gb] * a.layers
    else:
        sys.exit(__doc__)
    n_exp = a.experts or pack_ne or 384
    top_k = a.top_k or pack_k or 6
    hosts = json.load(open(a.hosts)) if a.hosts else FLEET
    hosts = [h for h in hosts if h["name"] not in a.drop]
    rows = None
    if a.popularity:
        try:
            rows = load_popularity(a.popularity, len(gb), n_exp)
        except PopError as e:
            sys.stderr.write("REFUSED: %s\n" % e)
            sys.exit(2)
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
    pop = None
    if rows is not None:
        pop = popularity_plan(a, hosts, gb, rows, n_exp, top_k, out)
        if a.perm_out:
            write_perm(a.perm_out, pop["perm"], n_exp)
        pop_perm = pop.pop("perm")
        if pop["remote_experts"] is None:            # flat: today's plan
            pop["remote_experts"] = spec
        res["popularity"] = pop
        res["remote_experts"] = pop["remote_experts"]
        short = 0.0                                  # the popularity plan placed (or refused) every expert itself
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print("experts: %.1f GB in %d layers" % (sum(gb), len(gb)))
        if pop is None:
            for r in out:
                print("  %-7s layers %s  %5.1f GB of %5.1f (%s-limited)" % (r["host"], ("%d-%d" % (r["first"], r["last"])) if r["layers"] else "-", r["gb"], r["cap_gb"], r["limit"]))
            print("  --remote-experts %s" % (spec or "(none)"))
            if short <= 1e-9:
                print("  experts alone, one token: %.1f ms (%.1f tok/s ceiling, layer after layer)" % (ms, 1000.0 / ms if ms else 0))
        else:
            print("popularity plan%s: %d experts a layer, %d picks, %s" % (" (flat file: today's layer-major plan)" if pop["flat"] else "", n_exp, top_k,
                                                                       ("%d layer group(s)" % pop["groups"]) if pop["groups"] else "no groups"))
            for hr in pop["hosts"]:
                print("  %-7s %6.1f GB  %5.1f%% of the hits  %s" % (hr["host"], hr["gb"], 100 * hr["hit_share"],
                                                                 " ".join("L%d-%d/E%d-%d" % (s["layers"][0], s["layers"][1], s["experts"][0], s["experts"][1]) for s in hr["segments"][:6])))
            if pop["ssd_hit_share"]:
                print("  conductor SSD tail: %.1f%% of the hits" % (100 * pop["ssd_hit_share"]))
            print("  --remote-experts %s" % (pop["remote_experts"] or "(none)"))
            if a.perm_out:
                print("  expert renumbering: %s (the conductor: STRATA_DS4_EXPERT_PERM; the slices: slice_stream.py emit --perm)" % a.perm_out)
            print("  experts alone, one token: %.1f ms (today's layer-major plan on the same hits: %s ms)" %
                  (pop["expected_ms_per_token"], pop["baseline_ms_per_token"] if pop["baseline_ms_per_token"] is not None else "does not fit"))
    if short > 1e-9:
        sys.stderr.write("REFUSED: %d layers (%.1f GB) have no host: free disk or memory on %s, or serve them from the conductor's SSD\n"
                         % (len(gb) - i, short, ", ".join(h["name"] for h in hosts)))
        sys.exit(3)


if __name__ == "__main__":
    main()
