#!/usr/bin/env python3
"""A model's geometry and byte budget from GGUF headers only: layers, experts, top-k, widths, and the bytes of every
tensor class, so a placement can be computed for a model nobody has downloaded.

Nothing is read from the weights.  The inputs are the GGUF headers (metadata + tensor directory) of every shard; the
size of a tensor is its block geometry (tools/gguf_reader.py BLOCK_GEOMETRY) when the type is known, else the distance
to the next tensor's offset (the last tensor of a shard: the shard's file size, which `--hf` takes from the repository
listing).  So a type this repository has no kernel for (Unsloth's sub-IQ1_S type 66) still has an exact byte count.

    python tools/gguf_geometry.py SHARD.gguf [SHARD.gguf ...]            # local files (full files or just their headers)
    python tools/gguf_geometry.py --hf unsloth/Qwen3.8-2.4T-A95B-GGUF/UD-IQ1_M [--cache DIR]
                                                                          # HTTP range requests for the HEADERS only
    ... [--json] [--model-spec OUT.json --name NAME --hidden-bytes 32768]  # xl.model_spec/1 for BigBrain/tools/xl/placement.py

`--hf` never downloads weights: shard 1 is read as a range of at most 64 MiB, every other shard as 4 MiB (the range
doubles up to 64 MiB only when a header does not fit); a bigger range is refused.  No token is read or sent.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import GGUFFile  # noqa: E402

MAX_RANGE = 64 << 20
FIRST_RANGE = 4 << 20
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


# ------------------------------------------------------------------ headers over HTTP (range requests only)
def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "strata-gguf-geometry"})
    if rng is not None:
        if rng[1] - rng[0] + 1 > MAX_RANGE:
            raise ValueError(f"refusing a range of {rng[1] - rng[0] + 1:,} bytes (limit {MAX_RANGE:,}): headers only")
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_headers(repo_path: str, cache: str) -> tuple[list[str], dict[str, int]]:
    """`owner/repo/subdir`: the shard headers of that subdirectory into `cache`; returns (paths in shard order, {basename: file size})."""
    parts = repo_path.strip("/").split("/")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    listing = json.loads(_get(f"https://huggingface.co/api/models/{repo}/tree/main/{sub}?recursive=1"))
    shards = sorted((e["path"], int(e["size"])) for e in listing if e["path"].endswith(".gguf"))
    if not shards:
        raise SystemExit(f"no .gguf files under {repo}/{sub}")
    os.makedirs(cache, exist_ok=True)
    out, sizes = [], {}
    for path, size in shards:
        name = os.path.basename(path)
        sizes[name] = size
        dst = os.path.join(cache, name)
        want = min(size, MAX_RANGE if path == shards[0][0] else FIRST_RANGE)
        while True:
            if not os.path.exists(dst) or os.path.getsize(dst) < min(want, size):
                with open(dst, "wb") as f:
                    f.write(_get(f"https://huggingface.co/{repo}/resolve/main/{path}", (0, want - 1)))
            try:
                GGUFFile(dst)
                break
            except ValueError as e:
                if "header ends early" not in str(e) or want >= min(size, MAX_RANGE):
                    raise
                want = min(want * 2, size, MAX_RANGE)
        out.append(dst)
    return out, sizes


# ------------------------------------------------------------------ the geometry
def tensor_sizes(g: GGUFFile, file_size: int | None) -> list[tuple]:
    """[(TensorInfo, nbytes or None)] for one shard.  nbytes: block geometry, else the distance to the next offset."""
    ts = sorted(g.tensors, key=lambda t: t.offset)
    out = []
    for i, t in enumerate(ts):
        n = t.expected_bytes()
        if n is None:
            if i + 1 < len(ts):
                n = ts[i + 1].offset - t.offset
            elif file_size is not None:
                n = file_size - g.data_start - t.offset
        out.append((t, n))
    return out


def analyse(paths: list[str], sizes: dict[str, int] | None = None) -> dict:
    sizes = sizes or {}
    gs = [GGUFFile(p) for p in paths]
    meta = {}
    for g in gs:   # the first shard carries the metadata; later shards only split.* keys
        for k, v in g.metadata.items():
            meta.setdefault(k, v)
    arch = meta.get("general.architecture", "?")
    pre = arch + "."

    def m(key, default=None):
        return meta.get(pre + key, default)

    n_blocks = m("block_count")
    n_mtp = m("nextn_predict_layers", 0) or 0
    n_trunk = n_blocks - n_mtp
    interval = m("full_attention_interval")
    geo = {
        "architecture": arch, "name": meta.get("general.name"), "size_label": meta.get("general.size_label"),
        "block_count": n_blocks, "trunk_layers": n_trunk, "mtp_layers": n_mtp,
        "n_embd": m("embedding_length"), "context_length": m("context_length"),
        "n_head": m("attention.head_count"), "n_head_kv": m("attention.head_count_kv"),
        "head_dim": m("attention.key_length"), "rope_dim": m("rope.dimension_count"),
        "n_expert": m("expert_count"), "n_expert_used": m("expert_used_count"),
        "expert_ff": m("expert_feed_forward_length"), "shared_expert_ff": m("expert_shared_feed_forward_length"),
        "full_attention_interval": interval,
        "ssm": {"conv_kernel": m("ssm.conv_kernel"), "state_size": m("ssm.state_size"), "k_heads": m("ssm.group_count"),
                "v_heads": m("ssm.time_step_rank"), "inner_size": m("ssm.inner_size")},
        "n_vocab": len(meta["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in meta else None,
    }
    if interval:
        geo["attention_layers"] = n_trunk // interval
        geo["gdn_layers"] = n_trunk - n_trunk // interval

    exp = collections.defaultdict(dict)          # layer -> {gate/up/down: (type, shape, nbytes)}
    dense_layer = collections.defaultdict(int)   # layer -> non-expert bytes in the layer
    dense_types = collections.defaultdict(collections.Counter)
    glob = {}
    unknown_size = []
    total = 0
    for p, g in zip(paths, gs):
        fs = sizes.get(os.path.basename(p), os.path.getsize(p))
        for t, n in tensor_sizes(g, fs):
            if n is None:
                unknown_size.append(t.name)
                continue
            total += n
            mo = EXPERT_RE.match(t.name)
            if mo:
                exp[int(mo.group(1))][mo.group(2)] = (t.type_name, list(t.shape), n)
                continue
            mb = BLK_RE.match(t.name)
            if mb:
                dense_layer[int(mb.group(1))] += n
                dense_types[int(mb.group(1))][t.type_name] += 1
            else:
                glob[t.name] = {"type": t.type_name, "shape": list(t.shape), "bytes": n}

    layers = []
    for l in sorted(exp):
        e = exp[l]
        per_exp = sum(v[2] for v in e.values()) // geo["n_expert"]
        layers.append({"layer": l, "kind": "mtp" if l >= n_trunk else "trunk",
                       "types": sorted({v[0] for v in e.values()}), "expert_tensor_bytes": sum(v[2] for v in e.values()),
                       "bytes_per_expert": per_exp, "dense_bytes": dense_layer[l]})
    trunk = [x for x in layers if x["kind"] == "trunk"]
    mtp = [x for x in layers if x["kind"] == "mtp"]
    gdn_b = {x["dense_bytes"] for x in trunk if not is_attn(x["layer"], interval)}
    att_b = {x["dense_bytes"] for x in trunk if is_attn(x["layer"], interval)}
    res = {
        "files": [os.path.basename(p) for p in paths], "geometry": geo,
        "expert_layers": layers,
        "trunk": {"layers": len(trunk), "types": sorted({t for x in trunk for t in x["types"]}),
                  "bytes_per_expert": sorted({x["bytes_per_expert"] for x in trunk}),
                  "expert_bytes": sum(x["expert_tensor_bytes"] for x in trunk)},
        "mtp": {"layers": len(mtp), "types": sorted({t for x in mtp for t in x["types"]}),
                "bytes_per_expert": sorted({x["bytes_per_expert"] for x in mtp}),
                "expert_bytes": sum(x["expert_tensor_bytes"] for x in mtp)},
        "dense": {"per_layer_gdn": sorted(gdn_b), "per_layer_attention": sorted(att_b),
                  "blk_total": sum(dense_layer.values()), "global": glob,
                  "global_total": sum(v["bytes"] for v in glob.values())},
        "tensor_bytes_total": total, "file_bytes_total": sum(sizes.get(os.path.basename(p), os.path.getsize(p)) for p in paths),
        "tensors_without_size": unknown_size,
    }
    res["expert_bytes_total"] = res["trunk"]["expert_bytes"] + res["mtp"]["expert_bytes"]
    return res


def is_attn(layer: int, interval: int | None) -> bool:
    return bool(interval) and layer % interval == interval - 1


def model_spec(a: dict, name: str, hidden_bytes: int | None, with_mtp: bool = True) -> dict:
    """xl.model_spec/1.  The MTP block is the last layer index (trunk_layers); its experts are a different (higher) quant, so with
    `with_mtp` the spec counts it as one more layer at the exact weighted-average bytes_per_expert (rounded up: the total never shrinks)."""
    geo = a["geometry"]
    n_exp = geo["n_expert"]
    layers = a["expert_layers"] if with_mtp else [x for x in a["expert_layers"] if x["kind"] == "trunk"]
    n_layers = len(layers)
    per_expert = -(-sum(x["expert_tensor_bytes"] for x in layers) // (n_layers * n_exp))
    blk_dense = sum(x["dense_bytes"] for x in layers)
    # the dense weights that always sit in VRAM on the conductor: every layer's attention / GDN / router / shared expert plus the output head
    # (token_embd is a lookup table kept in RAM; it is not in the per-layer VRAM figure)
    head = a["dense"]["global"].get("output.weight", {}).get("bytes", 0) + a["dense"]["global"].get("output_norm.weight", {}).get("bytes", 0)
    shared = -(-(blk_dense + head) // n_layers)
    return {"schema": "xl.model_spec/1", "name": name, "moe_layers": n_layers, "experts_per_layer": n_exp,
            "experts_per_token": geo["n_expert_used"], "bytes_per_expert": per_expert, "shared_bytes_per_layer": shared,
            "hidden_bytes": hidden_bytes or geo["n_embd"] * 4}


def gib(n: float) -> str:
    return f"{n / 2**30:,.2f} GiB"


def report(a: dict) -> str:
    g = a["geometry"]
    L = []
    L.append(f"{g['name']} ({g['architecture']})  files: {len(a['files'])} shards")
    L.append(f"  blocks {g['block_count']} = {g['trunk_layers']} trunk + {g['mtp_layers']} MTP; "
             f"{g.get('gdn_layers')} GDN + {g.get('attention_layers')} full attention (every {g['full_attention_interval']}th)")
    L.append(f"  n_embd {g['n_embd']}  heads {g['n_head']} / kv {g['n_head_kv']} x {g['head_dim']}  rope {g['rope_dim']}  vocab {g['n_vocab']}  ctx {g['context_length']}")
    s = g["ssm"]
    L.append(f"  GDN: conv {s['conv_kernel']} state {s['state_size']} k-heads {s['k_heads']} v-heads {s['v_heads']} inner {s['inner_size']}")
    L.append(f"  MoE: {g['n_expert']} experts/layer, top-{g['n_expert_used']} + 1 shared; expert ff {g['expert_ff']}, shared ff {g['shared_expert_ff']}")
    t, mt = a["trunk"], a["mtp"]
    L.append(f"  trunk experts: {'/'.join(t['types'])}  {t['bytes_per_expert']} B/expert  {gib(t['expert_bytes'])} ({t['layers']} layers)")
    if mt["layers"]:
        L.append(f"  MTP experts:   {'/'.join(mt['types'])}  {mt['bytes_per_expert']} B/expert  {gib(mt['expert_bytes'])} ({mt['layers']} layer)")
    d = a["dense"]
    L.append(f"  dense per layer: GDN {d['per_layer_gdn']}  attention {d['per_layer_attention']} B;  all blk dense {gib(d['blk_total'])}")
    for n, v in d["global"].items():
        L.append(f"  {n}: {v['type']} {v['shape']} {gib(v['bytes'])}")
    L.append(f"  experts total {gib(a['expert_bytes_total'])};  tensors total {gib(a['tensor_bytes_total'])};  files total {gib(a['file_bytes_total'])}")
    if a["tensors_without_size"]:
        L.append(f"  WARNING: {len(a['tensors_without_size'])} tensors without a known size (give --sizes or use --hf)")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*")
    ap.add_argument("--hf", help="owner/repo/subdir: read the shard headers by HTTP range request")
    ap.add_argument("--cache", default="gguf_headers", help="where --hf keeps the headers")
    ap.add_argument("--sizes", help="JSON {shard basename: file size} for header-only local files")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model-spec", metavar="OUT")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hidden-bytes", type=int, default=None)
    ap.add_argument("--no-mtp-layer", action="store_true", help="leave the in-file MTP block out of the model spec")
    a = ap.parse_args(argv)
    sizes = json.load(open(a.sizes)) if a.sizes else {}
    paths = list(a.shards)
    if a.hf:
        paths, sz = fetch_headers(a.hf, a.cache)
        sizes.update(sz)
    if not paths:
        ap.error("give shard files or --hf")
    res = analyse(paths, sizes)
    if a.model_spec:
        name = a.name or res["geometry"]["name"]
        with open(a.model_spec, "w") as f:
            json.dump(model_spec(res, name, a.hidden_bytes, not a.no_mtp_layer), f, indent=2)
            f.write("\n")
    print(json.dumps(res, indent=1) if a.json else report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
