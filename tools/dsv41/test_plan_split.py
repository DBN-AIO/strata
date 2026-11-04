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


# ---------------------------------------------------------------------------------------------------- BBXL S6: storage tiers

def mount(name, kind, free, reserve=0.0, path=None):
    return {"name": name, "kind": kind, "free_gb": free, "reserve_gb": reserve, "path": path or "/mnt/" + name.strip("/")}


# 8 layers of 1 GB.  disk_gb (the old single number) counts the SATA disks too: it is what plan_split used before the storage tiers.
def sfleet(rig_nvme=2.0, t340_tmpfs=3.0, rig_sata=50.0, stream=None):
    f = [
        {"name": "cond", "addr": None, "mem_gb": 3.0, "disk_gb": 1e9, "compute_ms": 2.5, "wire_ms": 0.0,
         "storage": [mount("nvmeA", "nvme", 2.0), mount("nvmeB", "nvme", 1.5), mount("bulk", "sata", 400.0)]},
        {"name": "rig", "addr": "10.0.0.2:7840", "mem_gb": 10.0, "disk_gb": rig_nvme + rig_sata, "compute_ms": 4.5, "wire_ms": 1.2,
         "storage": [mount("models", "nvme", rig_nvme), mount("root", "sata", rig_sata)]},
        {"name": "t340", "addr": "10.0.0.3:7840", "mem_gb": 10.0, "disk_gb": 53.0, "compute_ms": 6.4, "wire_ms": 0.75,
         "storage": [mount("shm", "tmpfs", t340_tmpfs), mount("root", "sata", 50.0)]},
    ]
    if stream is not None:
        f.append(stream)
    return f


class Storage(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def plan(self, fleet, *args):
        fp = os.path.join(self.d, "fleet.json")
        with open(fp, "w") as f:
            json.dump(fleet, f)
        return subprocess.run([sys.executable, PLAN, "--uniform-gb", "1.0", "--layers", str(NL), "--experts", str(NE), "--hosts", fp, "--json"]
                              + list(args), capture_output=True, text=True)

    def popfile(self):
        p = os.path.join(self.d, "pop.bin")
        write_profile(p, zipf_rows())
        return p

    def targets(self, res):
        return [t for t in res["storage"]["targets"]]

    def test_sata_is_never_chosen_while_nvme_has_room(self):
        r = self.plan(sfleet())
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        by = {p["host"]: p for p in res["plan"]}
        self.assertLessEqual(by["rig"]["gb"], 2.0 + 1e-9, "the rig's 50 GB of SATA is not an expert disk")
        self.assertEqual(res["storage"]["sata_used_gb"], 0)
        for t in self.targets(res):
            self.assertIn(t["kind"], ("nvme", "tmpfs"), t)

    def test_sata_refused_without_the_flag_and_labelled_with_it(self):
        f = sfleet(rig_nvme=1.0)                 # NVMe room 3 (the conductor's memory) + 1 + 3 = 7 < 8: SATA would be the only room
        r = self.plan(f)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("SATA", r.stderr)
        r = self.plan(f, "--allow-sata")
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        sata = [t for t in self.targets(res) if t["kind"] == "sata"]
        self.assertTrue(sata, "the last GB went to SATA")
        self.assertAlmostEqual(res["storage"]["sata_used_gb"], 1.0, places=3)   # whole layers: the 8th
        self.assertTrue(all(t["last_resort"] for t in sata))

    def test_allow_sata_does_not_use_sata_when_nvme_room_exists(self):
        r = self.plan(sfleet(), "--allow-sata")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["storage"]["sata_used_gb"], 0)

    def test_an_excluded_disk_is_never_used_even_with_allow_sata(self):
        stream = {"name": "stream", "addr": "10.0.0.4:7840", "mem_gb": 30.0, "disk_gb": 500.0, "compute_ms": 2.9, "wire_ms": 0.9,
                  "storage": [mount("C:", "nvme", 16.0, reserve=15.0), mount("D:", "excluded", 500.0)]}
        f = sfleet(rig_nvme=1.0, stream=stream)   # 1 GB usable on C:, 500 on the failing D:
        r = self.plan(f, "--allow-sata")
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        by = {p["host"]: p for p in res["plan"]}
        self.assertLessEqual(by["stream"]["gb"], 1.0 + 1e-9)
        for t in self.targets(res):
            self.assertNotEqual(t["mount"], "D:")

    def test_every_slice_names_its_mount_and_path(self):
        res = json.loads(self.plan(sfleet()).stdout)
        names = {(t["host"], t["mount"]) for t in self.targets(res)}
        self.assertIn(("rig", "models"), names)
        self.assertIn(("t340", "shm"), names)
        for t in self.targets(res):
            self.assertTrue(t["path"] and t["layers"] and t["gb"] > 0, t)
        # a host with two NVMe mounts splits by whole layers
        cond = [t for t in self.targets(res) if t["host"] == "cond"]
        self.assertEqual(sum(t["gb"] for t in cond), 3.0)
        self.assertEqual({t["mount"] for t in cond}, {"nvmeA", "nvmeB"})

    def test_popularity_plan_follows_the_same_rule(self):
        r = self.plan(sfleet(), "--popularity", self.popfile())
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout)
        self.assertEqual(res["storage"]["sata_used_gb"], 0)
        for t in self.targets(res):
            self.assertIn(t["kind"], ("nvme", "tmpfs"), t)
        by = {h["host"]: h for h in res["popularity"]["hosts"]}
        self.assertLessEqual(by["rig"]["gb"], 2.0 + 1e-9)
        r = self.plan(sfleet(rig_nvme=1.0), "--popularity", self.popfile())
        self.assertEqual(r.returncode, 3)
        self.assertIn("SATA", r.stderr)

    def test_the_storage_report_says_what_to_free(self):
        # no disk limit would put 3 / 3 / 2 GB; the rig has 1 GB of NVMe, the t340 has 2 GB of tmpfs
        f = sfleet(rig_nvme=1.0, t340_tmpfs=2.0)
        r = self.plan(f, "--storage-report")
        res = json.loads(r.stdout) if r.stdout.strip() else {}
        self.assertIn("storage_report", res, r.stderr)
        rep = {h["host"]: h for h in res["storage_report"]["hosts"]}
        self.assertGreater(rep["rig"]["free_needed_gb"], 0)
        self.assertEqual(rep["cond"]["free_needed_gb"], 0)
        self.assertGreater(res["storage_report"]["fleet_free_needed_gb"], 0)

    def test_bad_storage_entries_are_refused(self):
        for bad in ({"name": "x", "kind": "floppy", "free_gb": 1}, {"name": "x", "kind": "nvme", "free_gb": -1}, {"kind": "nvme", "free_gb": 1}):
            f = sfleet()
            f[1]["storage"] = [bad]
            r = self.plan(f)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("storage", r.stderr)


if __name__ == "__main__":
    unittest.main()
