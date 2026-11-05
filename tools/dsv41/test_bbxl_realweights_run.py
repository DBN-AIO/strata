#!/usr/bin/env python3
"""bbxl_realweights_run.py (BBXL S7): the arms, the preflight refusal, --dry-run, the RESULTS / JSON writer, and (with BUILD_DIR) the whole run on a synthetic pack.

    python3 tools/dsv41/test_bbxl_realweights_run.py [BUILD_DIR]     # BUILD_DIR = a build with ds4_net_test + strata-dsv41-expert-server: adds the end-to-end test
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "bbxl_realweights_run.py")
BUILD = sys.argv.pop(1) if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.environ.get("BUILD_DIR")
sys.path.insert(0, HERE)
import bbxl_realweights_run as R  # noqa: E402


def sh(args, **kw):
    return subprocess.run([sys.executable, RUN] + args, capture_output=True, text=True, **kw)


def manifest(tmp, floor=None):
    fleet = [{"name": "t340"}, {"name": "rig"}]
    man = {"engine": {"host": "t340", "bin": "/x/strata-dsv41", "pack": "/x/pack"}, "vocab": 1000, "runs": 1, "sizes": {"2k": 2048},
           "hosts": {"t340": {}, "rig": {"ssh": "u@rig", "os": "linux", "gpu": {"uuid": "GPU-abc", "min_free_mib": 3000}}},
           "plans": {"before": {"remote_experts": "rig:7840=0-39", "servers": [{"host": "rig", "bin": "/x/srv", "gpu_bin": "/x/gsrv", "pack": "/x/slice", "listen": "0.0.0.0:7840", "serve_layers": "0-39"}]}}}
    if floor:
        man["floors"] = {"t340": ["ram_gb", floor]}
    fp, mp = os.path.join(tmp, "fleet.json"), os.path.join(tmp, "man.json")
    json.dump(fleet, open(fp, "w"))
    json.dump(man, open(mp, "w"))
    return fp, mp


class Arms(unittest.TestCase):
    def test_each_stage_differs_from_base_by_one_thing(self):
        base = next(a for a in R.ARMS if a["id"] == "base")
        for a in R.ARMS:
            if a["id"] in ("base", "all"):
                continue
            diffs = [k for k in ("server", "plan", "flags") if a[k] != base[k]]
            diffs += [k for k in set(a["env"]) | set(base["env"]) if a["env"].get(k) != base["env"].get(k)]
            if a["id"] == "s3":   # plan after + the perm env: one change (the renumbering), two knobs
                self.assertEqual(sorted(diffs), ["_PERM", "plan"])
            else:
                self.assertEqual(len(diffs), 1, (a["id"], diffs))

    def test_floors_are_the_ordered_ones(self):
        self.assertEqual(R.FLOORS, {"gaming": ("ram_gb", 12.0), "t340": ("ram_gb", 16.0), "rig": ("ram_gb", 10.0), "stream": ("disk_c_gb", 15.0)})


class Preflight(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc/meminfo")
    def test_refuses_when_a_ram_floor_breaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp, mp = manifest(tmp, floor=10 ** 6)
            r = sh(["--manifest", mp, "--fleet", fp, "--out", os.path.join(tmp, "o"), "--dry-run"])
            # (dry-run never refuses: it prints the probes) - the real refusal needs the probes to run, so use the table check directly
            self.assertEqual(r.returncode, 0, r.stderr)
            rows, problems = R.preflight(R.Runner(False, tmp), [{"name": "t340"}], {"t340": ("ram_gb", 10 ** 6)}, {"t340"}, False)
            self.assertEqual(len(problems), 1)
            self.assertIn("floor", problems[0])
            rows, problems = R.preflight(R.Runner(False, tmp), [{"name": "t340"}], {"t340": ("ram_gb", 0.001)}, {"t340"}, False)
            self.assertEqual(problems, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc/meminfo")
    def test_exit_3_and_nothing_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp, mp = manifest(tmp, floor=10 ** 6)
            m = json.load(open(mp))
            m["hosts"]["rig"] = {}   # (no ssh, no gpu probe: only the t340 RAM floor is checked, on this machine)
            json.dump(m, open(mp, "w"))
            r = sh(["--manifest", mp, "--fleet", fp, "--out", os.path.join(tmp, "o")])
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            self.assertIn("REFUSED", r.stderr)
            self.assertNotIn("== servers", r.stdout)


class DryRun(unittest.TestCase):
    def test_prints_every_command_and_runs_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp, mp = manifest(tmp)
            r = sh(["--manifest", mp, "--fleet", fp, "--out", os.path.join(tmp, "o"), "--dry-run"])
            self.assertEqual(r.returncode, 0, r.stderr)
            o = r.stdout
            self.assertIn("nvidia-smi", o)
            self.assertIn("strata-dsv41", o)
            self.assertIn("expert-server-gpu", o)                     # the GPU arm's server
            self.assertIn("STRATA_DS4_NET_PREFILL=2", o)              # the S7 arm
            self.assertIn("--dspark auto", o)
            self.assertGreater(o.count("warm-up"), 3)
            self.assertTrue(all(ln.startswith(("[dry-run]", "==", "  ", "note:")) for ln in o.splitlines() if ln.strip()), o)
            self.assertFalse(os.path.exists(os.path.join(tmp, "o", "RESULTS.md")))


class Writer(unittest.TestCase):
    def test_results_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = [a for a in R.ARMS if a["id"] in ("base", "s7")]
            res = {("base", "2k"): dict(prefill=[100.0, 110.0, 105.0], decode=[10.0, 10.5, 10.2], hashes=["a", "b", "c"]),
                   ("s7", "2k"): dict(prefill=[200.0, 210.0, 205.0], decode=[10.0, 10.5, 10.2], hashes=["a", "b", "c"])}
            eng = R.RealEngine({"engine": {}})
            R.write_outputs(tmp, "2026-11-04T00:00:00", eng, [], [("rig", "ram_gb", 10.0, 13.0, True)], arms, ["s2"], {"2k": 2048}, 3, 128, res, {})
            md = open(os.path.join(tmp, "RESULTS.md")).read()
            self.assertIn("| S7 | s7 | 2k | 205.0 (200.0 210.0 205.0) | x1.95 |", md)
            self.assertIn("Not run: s2", md)
            rows = [json.loads(x) for x in open(os.path.join(tmp, "results.jsonl"))]
            self.assertEqual(len(rows), 4)
            r = next(x for x in rows if x["arm"] == "s7" and x["metric"] == "prefill")
            self.assertEqual((r["schema"], r["value"], r["unit"], r["prompt_tokens"], r["tokens_match_base"]), ("bbxl-bench/1", 205.0, "tok/s", 2048, True))

    def test_parsers(self):
        e = R.RealEngine({"engine": {}})
        err = "prompt: 2048 tokens in 4000 ms (512.00 tok/s, windows of 16)\ndecode: 128 tokens in 8000 ms (15.88 tok/s); CPU lane: 1 of 2"
        r = e.parse(0, "ids", err)
        self.assertEqual((r["prefill"], r["decode"]), (512.0, 15.88))
        self.assertIsNone(e.parse(1, "", err))
        self.assertIsNone(e.parse(0, "", "prompt: nothing"))


@unittest.skipUnless(BUILD and os.path.exists(os.path.join(BUILD, "ds4_net_test")), "needs BUILD_DIR with ds4_net_test (the end-to-end run on a synthetic pack)")
class Synthetic(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = os.path.join(tmp, "pack")
            subprocess.check_call([sys.executable, os.path.join(HERE, "make_synthetic_pack.py"), pack, "--seed", "1"], stdout=subprocess.DEVNULL)
            out = os.path.join(tmp, "out")
            r = sh(["--synthetic", BUILD, pack, "--out", out, "--runs", "2", "--sizes", "2k"], timeout=1800)
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            md = open(os.path.join(out, "RESULTS.md")).read()
            for arm in ("base", "s4", "s5", "s7"):
                self.assertIn("| %s |" % arm, md)
            rows = [json.loads(x) for x in open(os.path.join(out, "results.jsonl"))]
            self.assertTrue(all(x["schema"] == "bbxl-bench/1" and x["value"] > 0 and len(x["runs"]) == 2 for x in rows))
            # every arm's tokens equal base's (same toy, same rows whoever computed them)
            self.assertTrue(all(x["tokens_match_base"] in (True, None) for x in rows), [x for x in rows if x["tokens_match_base"] is False])


if __name__ == "__main__":
    unittest.main()
