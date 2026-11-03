#!/usr/bin/env python3
"""plan_split.py: popularity-aware placement (BBXL S3) and the storage tiers (BBXL S6).  python3 tools/dsv41/test_plan_split.py

The popularity file is the engine's own expert profile (`strata-dsv41 --expert-profile FILE --profile-run IDS`): raw uint32 hit
counts, [layers (+ up to 8 drafter rows)][experts], or a TSV `layer<TAB>expert<TAB>count`.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "plan_split.py")
NL, NE = 8, 32

# a small fleet: the conductor holds 3 GB (1 of them VRAM), two remotes 3 GB each; 8 layers of 1 GB do not all fit on the conductor
FLEET = [
    {"name": "cond", "addr": None, "mem_gb": 3.0, "vram_gb": 1.0, "disk_gb": 1e9, "compute_ms": 2.5, "wire_ms": 0.0},
    {"name": "rig", "addr": "10.0.0.2:7840", "mem_gb": 3.0, "disk_gb": 3.0, "compute_ms": 4.5, "wire_ms": 1.2},
    {"name": "t340", "addr": "10.0.0.3:7840", "mem_gb": 3.0, "disk_gb": 3.0, "compute_ms": 6.4, "wire_ms": 0.75},
]


def zipf_rows(s=1.1, seed=3):
    rng = random.Random(seed)
    rows = []
    for _ in range(NL):
        order = list(range(NE))
        rng.shuffle(order)
        w = [0] * NE
        for rank, e in enumerate(order):
            w[e] = int(1e6 / (rank + 1) ** s)
        rows.append(w)
    return rows


def write_profile(path, rows, extra=0):
    with open(path, "wb") as f:
        for r in rows + [[0] * NE] * extra:
            f.write(struct.pack("<%dI" % NE, *r))


class PlanSplit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = self.td.name
        self.fleet = os.path.join(self.d, "fleet.json")
        json.dump(FLEET, open(self.fleet, "w"))

    def tearDown(self):
        self.td.cleanup()

    def run_plan(self, *args):
        return subprocess.run([sys.executable, PLAN, "--uniform-gb", "1.0", "--layers", str(NL), "--experts", str(NE),
                               "--hosts", self.fleet, "--json"] + list(args), capture_output=True, text=True)

    def pop(self, rows, name="pop.bin", extra=0):
        p = os.path.join(self.d, name)
        write_profile(p, rows, extra)
        return p

    def test_skewed_hot_experts_go_to_the_fastest_tier(self):
        rows = zipf_rows()
        perm = os.path.join(self.d, "perm.bin")
        r = self.run_plan("--popularity", self.pop(rows), "--perm-out", perm)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)["popularity"]
        # the conductor holds 3 of 8 GB = 37.5 % of the experts but must serve far more of the hits
        self.assertGreater(res["hosts"][0]["hit_share"], 0.55)
        self.assertLess(res["expected_ms_per_token"], res["baseline_ms_per_token"])
        # the hottest expert of every layer has rank 0 (the conductor's own, never sent over the wire)
        blob = open(perm, "rb").read()
        self.assertEqual(blob[:8], b"DS4PERM1")
        nl, ne = struct.unpack_from("<II", blob, 8)
        self.assertEqual((nl, ne), (NL, NE))
        for il in range(NL):
            ranks = struct.unpack_from("<%dH" % NE, blob, 16 + il * NE * 2)
            self.assertEqual(sorted(ranks), list(range(NE)), "a permutation")
            hottest = max(range(NE), key=lambda e: rows[il][e])
            self.assertEqual(ranks[hottest], 0)

    def test_route_ranges_do_not_overlap_and_the_conductor_keeps_the_front(self):
        r = self.run_plan("--popularity", self.pop(zipf_rows()))
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        cover = {}
        for item in res["remote_experts"].split(","):
            addr, rest = item.split("=")
            lay, _, ex = rest.partition("/")
            lo, hi = map(int, lay.split("-"))
            e0, e1 = map(int, ex.split("-"))
            self.assertGreater(e0, 0, "rank 0.. belongs to the conductor")
            for il in range(lo, hi + 1):
                for e in range(e0, e1 + 1):
                    self.assertNotIn((il, e), cover)
                    cover[(il, e)] = addr

    def test_uniform_popularity_gives_todays_plan(self):
        base = self.run_plan()
        flat = self.run_plan("--popularity", self.pop([[100] * NE for _ in range(NL)]))
        self.assertEqual(base.returncode, 0, base.stderr)
        self.assertEqual(flat.returncode, 0, flat.stderr)
        self.assertEqual(json.loads(flat.stdout)["remote_experts"], json.loads(base.stdout)["remote_experts"])
        self.assertEqual(json.loads(flat.stdout)["plan"], json.loads(base.stdout)["plan"])

    def test_malformed_files_are_refused(self):
        good = zipf_rows()
        cases = {}
        p = self.pop(good, "trunc.bin")
        open(p, "r+b").truncate(os.path.getsize(p) - 7)
        cases["truncated"] = p
        cases["too long"] = self.pop(good, "long.bin", extra=9)
        cases["zero"] = self.pop([[0] * NE for _ in range(NL)], "zero.bin")
        q = os.path.join(self.d, "strp.bin")
        open(q, "wb").write(b"STRP" + b"\0" * 4000)
        cases["ranked profile of the other engine"] = q
        t = os.path.join(self.d, "neg.tsv")
        open(t, "w").write("0\t1\t-5\n")
        cases["negative count"] = t
        t2 = os.path.join(self.d, "oor.tsv")
        open(t2, "w").write("0\t%d\t5\n" % NE)
        cases["expert out of range"] = t2
        for name, path in cases.items():
            r = self.run_plan("--popularity", path)
            self.assertEqual(r.returncode, 2, name + ": " + r.stdout + r.stderr)
            self.assertIn("popularity", r.stderr, name)
        r = self.run_plan("--popularity", os.path.join(self.d, "missing.bin"))
        self.assertEqual(r.returncode, 2)

    def test_tsv_equals_binary(self):
        rows = zipf_rows()
        t = os.path.join(self.d, "p.tsv")
        with open(t, "w") as f:
            for il, row in enumerate(rows):
                for e, c in enumerate(row):
                    f.write("%d\t%d\t%d\n" % (il, e, c))
        a = self.run_plan("--popularity", self.pop(rows))
        b = self.run_plan("--popularity", t)
        self.assertEqual(json.loads(a.stdout)["popularity"], json.loads(b.stdout)["popularity"])

    def test_drafter_rows_are_ignored(self):
        a = self.run_plan("--popularity", self.pop(zipf_rows()))
        b = self.run_plan("--popularity", self.pop(zipf_rows(), "x.bin", extra=3))
        self.assertEqual(json.loads(a.stdout)["remote_experts"], json.loads(b.stdout)["remote_experts"])

    def test_capacity_still_refused(self):
        tight = [dict(h) for h in FLEET]
        for h in tight:
            if h["addr"]:
                h["mem_gb"] = h["disk_gb"] = 1.0
        json.dump(tight, open(self.fleet, "w"))
        r = self.run_plan("--popularity", self.pop(zipf_rows()))
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
