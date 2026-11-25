#!/usr/bin/env python3
"""Plan a remote-experts split: which layers each expert server holds, and the exact commands to start it.

docs/REMOTE_EXPERTS.md describes the flags; this tool does the arithmetic a first-time user would otherwise do by hand.
The conductor keeps the first layers (and attention, router, shared expert); every server takes a block of the LAST
layers, as many as its RAM holds (after a headroom), in the order the hosts are given.

    python tools/remote_experts_plan.py --shards MODEL-00001-of-0000N.gguf [...] \
        --host rig=192.0.2.10:7840:24 --host box2=192.0.2.11:7840:48 [--headroom 0.15] [--json]
    python tools/remote_experts_plan.py --layers 48 --layer-gib 0.71 --host rig=192.0.2.10:7840:24    # no files

--host NAME=ADDRESS:PORT:RAM_GB (RAM_GB = memory the server may use for experts).  Nothing is started or changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GIB = 1 << 30


def parse_host(text: str) -> dict:
    """NAME=ADDRESS:PORT:RAM_GB -> dict.  ADDRESS may be [v6]; every part is checked, a typo is an error."""
    name, sep, rest = text.partition("=")
    if not sep or not name or not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"host {text!r}: expected NAME=ADDRESS:PORT:RAM_GB")
    addr, _, ram = rest.rpartition(":")
    addr, _, port = addr.rpartition(":")
    if not addr or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"host {text!r}: the port must be 1..65535")
    try:
        ram_gb = float(ram)
    except ValueError:
        raise ValueError(f"host {text!r}: RAM_GB must be a number") from None
    if ram_gb <= 0:
        raise ValueError(f"host {text!r}: RAM_GB must be above 0")
    return {"name": name, "address": addr, "port": int(port), "ram_gb": ram_gb}


def plan(layer_bytes: list[int], hosts: list[dict], headroom: float = 0.15) -> dict:
    """layer_bytes[i] = expert bytes of trunk layer i.  Servers fill from the last layer backwards, one block each."""
    if not layer_bytes:
        raise ValueError("the model has no expert layers")
    if not 0 <= headroom < 1:
        raise ValueError("headroom must be 0 or more and below 1")
    if len({h["name"] for h in hosts}) != len(hosts):
        raise ValueError("host names must be different")
    n = len(layer_bytes)
    end = n                      # exclusive end of the layers still unassigned
    servers = []
    for h in hosts:
        budget = h["ram_gb"] * GIB * (1 - headroom)
        used, start = 0.0, end
        while start > 0 and used + layer_bytes[start - 1] <= budget:
            start -= 1
            used += layer_bytes[start]
        got = end - start
        servers.append({**h, "first": start if got else None, "last": end - 1 if got else None,
                        "layers": got, "gib": round(used / GIB, 2)})
        end = start
    local = sum(layer_bytes[:end])
    return {"layers": n, "servers": servers, "conductor_layers": f"0-{end - 1}" if end else "none",
            "conductor_gib": round(local / GIB, 2), "headroom": headroom,
            "complete": end == 0, "unplaced_hosts": [s["name"] for s in servers if s["layers"] == 0]}


def commands(p: dict) -> list[str]:
    out = ["# on every host and on the conductor: the same secret (see docs/REMOTE_EXPERTS.md, Security)",
           "export STRATA_EXPERT_TOKEN=<your secret>", ""]
    spec = []
    for s in p["servers"]:
        if not s["layers"]:
            out.append(f"# {s['name']}: its RAM holds no whole layer, not used")
            continue
        r = f"{s['first']}-{s['last']}"
        out += [f"# {s['name']}: layers {r}, {s['gib']} GiB of experts",
                f"strata --pack PACK --native SHARD1.gguf --pcie-frac 0 --expert-server --listen {s['address']}:{s['port']} --serve-layers {r}"]
        spec.append(f"{s['address']}:{s['port']}={r}")
    if spec:
        out += ["", f"# conductor: runs layers {p['conductor_layers']} itself ({p['conductor_gib']} GiB of experts)",
                "strata ... --pcie-frac 0 --remote-experts " + ",".join(spec)]
    if not p["complete"]:
        out.append(f"# NOTE: the conductor keeps layers {p['conductor_layers']}; give the servers more RAM to hand out more.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", help="GGUF files of the model (headers are read, weights are not)")
    ap.add_argument("--layers", type=int, help="without files: number of expert layers")
    ap.add_argument("--layer-gib", type=float, help="without files: GiB of experts in one layer")
    ap.add_argument("--host", action="append", default=[], help="NAME=ADDRESS:PORT:RAM_GB, repeat per expert server")
    ap.add_argument("--headroom", type=float, default=0.15, help="share of each server's RAM kept free (default 0.15)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        if not a.host:
            raise ValueError("give at least one --host")
        hosts = [parse_host(h) for h in a.host]
        if a.shards:
            import gguf_geometry
            res = gguf_geometry.analyse(a.shards)
            layer_bytes = [x["expert_tensor_bytes"] for x in res["expert_layers"] if x["kind"] == "trunk"]
        elif a.layers and a.layer_gib:
            layer_bytes = [int(a.layer_gib * GIB)] * a.layers
        else:
            raise ValueError("give --shards, or --layers and --layer-gib")
        p = plan(layer_bytes, hosts, a.headroom)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(p, indent=2) if a.json else "\n".join(commands(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
