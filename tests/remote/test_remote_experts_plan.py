"""tools/remote_experts_plan.py: host parsing, the layer blocks, the printed commands."""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import remote_experts_plan as P  # noqa: E402

GIB = 1 << 30


class Hosts(unittest.TestCase):
    def test_good(self):
        h = P.parse_host("rig=192.0.2.10:7840:24")
        self.assertEqual((h["name"], h["address"], h["port"], h["ram_gb"]), ("rig", "192.0.2.10", 7840, 24.0))

    def test_ipv6(self):
        self.assertEqual(P.parse_host("a=[2001:db8::1]:7841:8")["address"], "[2001:db8::1]")

    def test_typos_are_errors(self):
        for bad in ("rig", "=192.0.2.1:7840:8", "rig=192.0.2.1:0:8", "rig=192.0.2.1:70000:8", "rig=192.0.2.1:78x0:8",
                    "rig=192.0.2.1:7840:lots", "rig=192.0.2.1:7840:0", "rig=:7840:8"):
            with self.assertRaises(ValueError, msg=bad):
                P.parse_host(bad)


class Plan(unittest.TestCase):
    def hosts(self, *rams):
        return [P.parse_host(f"h{i}=192.0.2.{i + 1}:{7840 + i}:{r}") for i, r in enumerate(rams)]

    def test_tail_block_fits_ram_after_headroom(self):
        # 48 layers of 1 GiB; 10 GB at 15% headroom = 8.5 GiB -> 8 whole layers: 40-47
        p = P.plan([GIB] * 48, self.hosts(10))
        s = p["servers"][0]
        self.assertEqual((s["first"], s["last"], s["layers"]), (40, 47, 8))
        self.assertEqual(p["conductor_layers"], "0-39")
        self.assertFalse(p["complete"])

    def test_two_servers_do_not_overlap_and_fill_backwards(self):
        p = P.plan([GIB] * 48, self.hosts(10, 20), headroom=0)
        a, b = p["servers"]
        self.assertEqual((a["first"], a["last"]), (38, 47))
        self.assertEqual((b["first"], b["last"]), (18, 37))
        self.assertEqual(p["conductor_layers"], "0-17")

    def test_everything_handed_out(self):
        p = P.plan([GIB] * 4, self.hosts(100))
        self.assertTrue(p["complete"])
        self.assertEqual(p["conductor_layers"], "none")
        self.assertEqual(p["servers"][0]["layers"], 4)

    def test_host_too_small_for_a_layer_is_skipped(self):
        p = P.plan([20 * GIB] * 4, self.hosts(10, 100))
        self.assertEqual(p["unplaced_hosts"], ["h0"])
        self.assertEqual(p["servers"][1]["layers"], 4)
        serve = [c for c in P.commands(p) if "--serve-layers" in c]
        self.assertEqual(len(serve), 1)
        self.assertIn("--listen 192.0.2.2:7841", serve[0])

    def test_uneven_layers(self):
        p = P.plan([GIB, GIB, 5 * GIB, GIB], self.hosts(3), headroom=0)   # 3 GiB: layer 3 fits, layer 2 (5 GiB) does not
        self.assertEqual((p["servers"][0]["first"], p["servers"][0]["last"]), (3, 3))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            P.plan([], self.hosts(8))
        with self.assertRaises(ValueError):
            P.plan([GIB], self.hosts(8), headroom=1)
        with self.assertRaises(ValueError):
            P.plan([GIB], self.hosts(8)[:1] * 2)

    def test_commands_match_the_documented_flags(self):
        p = P.plan([GIB] * 48, self.hosts(10), headroom=0)
        text = "\n".join(P.commands(p))
        self.assertIn("--expert-server --listen 192.0.2.1:7840 --serve-layers 38-47", text)
        self.assertIn("--remote-experts 192.0.2.1:7840=38-47", text)
        self.assertIn("STRATA_EXPERT_TOKEN", text)


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "remote_experts_plan.py"), *args],
                              capture_output=True, text=True)

    def test_json(self):
        r = self.run_cli("--layers", "48", "--layer-gib", "1", "--host", "rig=192.0.2.10:7840:12", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["servers"][0]["layers"], 10)

    def test_errors_exit_2(self):
        for args in ([], ["--host", "rig=192.0.2.1:7840:8"], ["--layers", "4", "--layer-gib", "1", "--host", "bad"]):
            self.assertEqual(self.run_cli(*args).returncode, 2, args)

    def test_shards_use_trunk_layers_only(self):
        """The MTP layer is not handed out: only kind == trunk counts (analyse() output faked, no files read)."""
        import io
        import types
        from contextlib import redirect_stdout
        layers = [{"layer": i, "kind": "trunk", "expert_tensor_bytes": GIB} for i in range(4)]
        layers.append({"layer": 4, "kind": "mtp", "expert_tensor_bytes": GIB})
        fake = types.SimpleNamespace(analyse=lambda paths: {"expert_layers": layers})
        sys.modules["gguf_geometry"] = fake
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = P.main(["--shards", "x.gguf", "--host", "rig=192.0.2.10:7840:100", "--json"])
        finally:
            sys.modules.pop("gguf_geometry", None)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["layers"], 4)


if __name__ == "__main__":
    unittest.main()
