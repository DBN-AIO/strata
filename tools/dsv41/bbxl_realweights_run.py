#!/usr/bin/env python3
"""BBXL combined real-weights run (S1-S7): ONE command, warm-up first, RESULTS table + dashboard JSON rows.

    python3 tools/dsv41/bbxl_realweights_run.py --manifest run.json [--out DIR] [--dry-run]

What it does, in order (docs/DEEPSEEK41_SPLIT.md "The combined real-weights run (BBXL S7)"):
  1. loads the hosts from tools/dsv41/fleet.json (S6) and the site paths from the manifest (tools/dsv41/bbxl_run_manifest.example.json);
  2. PREFLIGHT, and refuses (exit 3) when a RAM floor breaks (gaming >= 12 GB free, T340 >= 16, rig >= 10, stream C: >= 15 GB free) or a GPU lacks
     the VRAM the manifest asks for (nvidia-smi, by UUID);
  3. the arms below, grouped by (server kind, plan) so the servers restart as seldom as possible: servers up, a WARM-UP engine run per arm (not counted),
     then --runs timed runs at every prompt size (2k and 16k), each run with its own UNIQUE PREFIX (so no cache of an earlier run answers it);
  4. writes DIR/RESULTS.md (a table to paste into the docs), DIR/results.jsonl (one dashboard row per arm x prompt x metric, schema in the docs)
     and every raw log; stops every server it started, also on an error.

Arms: base = everything off (CPU servers, the plan without popularity, serial calls, --dspark none, the prompt through the conductor's own flow);
each stage arm changes ONE thing against base (before = base, after = the arm); `all` turns every one on.
    s1 DSpark auto (the split rule)      s2 GPU expert servers        s3 popularity renumbering (plan "after" + STRATA_DS4_EXPERT_PERM)
    s4 parallel remote calls             s5 prompt window path through the servers (STRATA_DS4_NET_PREFILL=1)
    s7 chunked flow through the servers (STRATA_DS4_NET_PREFILL=2)         all

--dry-run prints every command (preflight, servers, engine runs) and runs none.  --synthetic BUILD_DIR PACK runs the same machinery end to end on a synthetic
pack (ds4_net_test as the engine, expert servers on loopback); it never touches real weights.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "bbxl-bench/1"

# the floors (Captain 1 #22865): free RAM in GB on the host (MemAvailable: reclaimable cache counts), or free disk GB of a drive for the stream PC
FLOORS = {"gaming": ("ram_gb", 12.0), "t340": ("ram_gb", 16.0), "rig": ("ram_gb", 10.0), "stream": ("disk_c_gb", 15.0)}

# arm: server kind (cpu|gpu), plan (before|after), env, conductor flags.  Every stage arm differs from base by one thing.
BASE_ENV = {"STRATA_NET_SERIAL": "1"}
ARMS = [
    dict(id="base", stage="baseline", server="cpu", plan="before", env=dict(BASE_ENV), flags=["--dspark", "none"],
         what="CPU servers, plan without popularity, serial calls, --dspark none, the prompt through the conductor's own flow"),
    dict(id="s1", stage="S1", server="cpu", plan="before", env=dict(BASE_ENV), flags=["--dspark", "auto"], what="base + --dspark auto (the split rule keeps the drafter)"),
    dict(id="s2", stage="S2", server="gpu", plan="before", env=dict(BASE_ENV), flags=["--dspark", "none"], what="base + GPU expert servers (--expert-server-gpu)"),
    dict(id="s3", stage="S3", server="cpu", plan="after", env=dict(BASE_ENV, _PERM="1"), flags=["--dspark", "none"], what="base + experts renumbered by popularity (plan 'after')"),
    dict(id="s4", stage="S4", server="cpu", plan="before", env={}, flags=["--dspark", "none"], what="base + parallel remote calls (STRATA_NET_SERIAL unset)"),
    dict(id="s5", stage="S5", server="cpu", plan="before", env=dict(BASE_ENV, STRATA_DS4_NET_PREFILL="1"), flags=["--dspark", "none"],
         what="base + the prompt through the window path over the servers (STRATA_DS4_NET_PREFILL=1)"),
    dict(id="s7", stage="S7", server="cpu", plan="before", env=dict(BASE_ENV, STRATA_DS4_NET_PREFILL="2"), flags=["--dspark", "none"],
         what="base + the chunked flow through the servers (STRATA_DS4_NET_PREFILL=2)"),
    dict(id="all", stage="all", server="gpu", plan="after", env=dict(_PERM="1", STRATA_DS4_NET_PREFILL="2"), flags=["--dspark", "auto"],
         what="every stage on"),
]
SIZES = {"2k": 2048, "16k": 16384}


class Fail(Exception):
    pass


# ---------------------------------------------------------------- running things (local, over ssh, or only printed)
class Runner:
    def __init__(self, dry, logdir):
        self.dry, self.logdir = dry, logdir
        self.spawned = []

    @staticmethod
    def _wrap(host, argv, env, shell_prefix=""):
        """The argv the local machine runs for `argv` on `host` (a manifest host: {"ssh": "user@ip"} or local)."""
        e = {k: v for k, v in (env or {}).items() if not k.startswith("_")}
        ssh = host.get("ssh")
        if ssh:
            inner = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in e.items())
            inner = (shell_prefix + (inner + " " if inner else "") + shlex.join(argv))
            return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, inner], None
        env2 = dict(os.environ)
        env2.update(e)
        if (env or {}).get("_TOKEN_FILE"):   # the conductor's own token file: read here, handed to the engine, never printed
            env2["STRATA_EXPERT_TOKEN"] = open(env["_TOKEN_FILE"], encoding="utf-8").read().strip()
        return list(argv), env2

    def shown(self, host, argv, env):
        e = " ".join("%s=%s" % (k, v) for k, v in (env or {}).items() if not k.startswith("_"))
        return "%s: %s%s" % (host.get("name", "local"), (e + " ") if e else "", shlex.join(argv))

    def run(self, host, argv, env=None, timeout=7200, log=None):
        print(("[dry-run] " if self.dry else "$ ") + self.shown(host, argv, env), flush=True)
        if self.dry:
            return 0, "", ""
        cmd, env2 = self._wrap(host, argv, env)
        r = subprocess.run(cmd, capture_output=True, text=True, env=env2, timeout=timeout)
        if log:
            with open(os.path.join(self.logdir, log), "w", encoding="utf-8") as f:
                f.write("$ " + self.shown(host, argv, env) + "\n--- stdout\n" + r.stdout + "\n--- stderr\n" + r.stderr)
        return r.returncode, r.stdout, r.stderr

    def spawn(self, host, argv, env=None, log="server.log", token_file=None):
        """A server that stays up: returns a handle with stop()."""
        print(("[dry-run] " if self.dry else "$ ") + self.shown(host, argv, env) + "  &", flush=True)
        h = dict(host=host, pid=None, proc=None)
        if self.dry:
            return h
        logp = os.path.join(self.logdir, log)
        pre = 'STRATA_EXPERT_TOKEN="$(cat %s)" ' % shlex.quote(token_file) if token_file else ""
        if host.get("ssh"):
            cmd, _ = self._wrap(host, argv, env, shell_prefix=pre)
            cmd[-1] = "nohup nice -n 19 sh -c %s > %s 2>&1 < /dev/null & echo $!" % (shlex.quote("exec " + cmd[-1]), shlex.quote(host.get("remote_log_dir", "/var/tmp") + "/" + log))
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not r.stdout.strip().isdigit():
                raise Fail("cannot start a server on %s: %s" % (host["name"], r.stderr.strip()[-200:]))
            h["pid"] = int(r.stdout.strip())
        else:
            cmd, env2 = self._wrap(host, argv, env)
            if token_file:
                env2["STRATA_EXPERT_TOKEN"] = open(token_file, encoding="utf-8").read().strip()
            h["proc"] = subprocess.Popen(cmd, env=env2, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
        self.spawned.append(h)
        return h

    def stop(self, h):
        if self.dry or h is None:
            return
        host = h["host"]
        if h.get("proc") is not None:
            if h["proc"].poll() is None:
                h["proc"].terminate()
                try:
                    h["proc"].wait(timeout=15)
                except Exception:
                    h["proc"].kill()
        elif h.get("pid"):
            subprocess.run(["ssh", "-o", "BatchMode=yes", host["ssh"], "kill %d" % h["pid"]], capture_output=True, text=True, timeout=30)
        if h in self.spawned:
            self.spawned.remove(h)

    def stop_all(self):
        for h in list(self.spawned):
            try:
                self.stop(h)
            except Exception as e:  # never leave the others running because one stop failed
                print("stop failed: %s" % e, file=sys.stderr)


def wait_port(addr, timeout, dry):
    if dry:
        return True
    host, port = addr.rsplit(":", 1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, int(port)), timeout=3):
                return True
        except OSError:
            time.sleep(2)
    return False


# ---------------------------------------------------------------- preflight
def probe_ram_gb(run, host):
    if host.get("os") == "windows":
        cmd = ["powershell", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"]
    else:
        cmd = ["awk", "/MemAvailable/ {printf \"%.1f\", $2/1048576}", "/proc/meminfo"]
    rc, out, err = run.run(host, cmd)
    return float(out.strip()) if rc == 0 and out.strip() else None


def probe_disk_c_gb(run, host):
    cmd = ["powershell", "-NoProfile", "-Command", "[math]::Round((Get-PSDrive C).Free/1GB,1)"]
    rc, out, err = run.run(host, cmd)
    return float(out.strip()) if rc == 0 and out.strip() else None


def probe_gpu_free_mib(run, host):
    rc, out, err = run.run(host, ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"])
    d = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 2 and p[1].isdigit():
            d[p[0]] = int(p[1])
    return d if rc == 0 else None


def preflight(run, hosts, floors, used_hosts, dry):
    """Returns (rows, problems).  Hosts not used by the manifest are not probed."""
    rows, problems = [], []
    for h in hosts:
        if h["name"] not in used_hosts:
            continue
        kind, floor = floors.get(h["name"], (None, None))
        if kind == "ram_gb":
            v = probe_ram_gb(run, h)
        elif kind == "disk_c_gb":
            v = probe_disk_c_gb(run, h)
        else:
            v = None
        if kind:
            ok = dry or (v is not None and v >= floor)
            rows.append((h["name"], kind, floor, v, ok))
            if not ok:
                problems.append("%s: %s %s, the floor is %.0f" % (h["name"], kind, "unreadable" if v is None else "%.1f" % v, floor))
        g = h.get("gpu")
        if g:
            free = probe_gpu_free_mib(run, h)
            have = None if free is None else free.get(g["uuid"])
            ok = dry or (have is not None and have >= g["min_free_mib"])
            rows.append((h["name"], "gpu %s free MiB" % g["uuid"][:12], g["min_free_mib"], have, ok))
            if not ok:
                problems.append("%s: GPU %s has %s MiB free, the run needs %d" % (h["name"], g["uuid"], have, g["min_free_mib"]))
    return rows, problems


# ---------------------------------------------------------------- the engines
class RealEngine:
    """strata-dsv41 on the conductor host: prompt: / decode: lines on stderr (ds4_main.cpp)."""
    name = "strata-dsv41"

    def __init__(self, man):
        self.m = man

    def argv(self, arm, plan, ids_file, n_decode):
        e = self.m["engine"]
        a = [e["bin"], "--pack", plan.get("conductor_pack", e["pack"]), "--remote-experts", plan["remote_experts"], "--ids", "@" + ids_file, "--n", str(n_decode), "--verbose"]
        a += arm["flags"] + e.get("extra_flags", [])
        if arm["env"].get("_PERM") and plan.get("profile"):
            a += ["--expert-profile", plan["profile"]]
        return a

    def env(self, arm, plan):
        env = dict(arm["env"])
        if env.pop("_PERM", None) and plan.get("perm"):
            env["STRATA_DS4_EXPERT_PERM"] = plan["perm"]
        if self.m["engine"].get("token_file"):
            env["_TOKEN_FILE"] = self.m["engine"]["token_file"]
        return env

    PRE = re.compile(r"prompt: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s")
    DEC = re.compile(r"decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s")

    def parse(self, rc, out, err):
        p, d = self.PRE.search(err), self.DEC.search(err)
        if rc != 0 or not p or not d:
            return None
        return dict(prefill=float(p.group(3)), decode=float(d.group(3)), tokens_hash=str(zlib.crc32(out.strip().encode())))

    def write_ids(self, base_ids, n, salt, path):
        """The run's prompt: its own unique 16-token prefix, then the base ids, cut to n tokens (a prefix cache cannot answer it)."""
        vocab = self.m.get("vocab", 100000)
        pre = [(salt * 7919 + i * 104729 + 17) % vocab for i in range(16)]
        ids = (pre + base_ids * (n // max(1, len(base_ids)) + 1))[:n]
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(str(i) for i in ids))


class SyntheticEngine(RealEngine):
    """ds4_net_test on a synthetic pack, the same arms mapped to its switches (what each arm means is in the docs)."""
    name = "ds4_net_test (synthetic)"
    MAP = {  # arm id -> (ds4_net_test flags, env); None = not measurable without real weights / a GPU
        "base": (["--mode", "local", "--threads", "4"], {}),
        "s1": None, "s2": None, "s3": None, "all": None,
        "s4": (["--mode", "remote", "--prompt-path", "--overlap"], {}),
        "s5": (["--mode", "remote", "--prompt-path"], {}),
        "s7": (["--mode", "remote", "--chunk-flow"], {}),
    }

    def __init__(self, man, build):
        super().__init__(man)
        self.build = build

    def supported(self, arm):
        return self.MAP.get(arm["id"]) is not None

    def argv(self, arm, plan, ids_file, n_decode):
        flags, _ = self.MAP[arm["id"]]
        salt = int(re.search(r"-s(\d+)\.ids$", ids_file).group(1)) if re.search(r"-s(\d+)\.ids$", ids_file) else 0
        n = len(open(ids_file).read().split(","))
        a = [os.path.join(self.build, "ds4_net_test"), "--pack", self.m["engine"]["pack"], "--prompt", str(n), "--prompt-salt", str(salt), "--tokens", str(n_decode), "--time", "--slab", "512"]
        a += flags
        if "remote" in flags:
            a += ["--remote", plan["remote_experts"]]
        if "--chunk-flow" not in flags and "--prompt-path" not in flags and "local" in flags and n > 16:
            pass
        return a

    def env(self, arm, plan):
        e = dict(self.MAP[arm["id"]][1])
        e["STRATA_EXPERT_TOKEN"] = "bbxl-synthetic"
        if "STRATA_NET_SERIAL" in arm["env"]:
            e["STRATA_NET_SERIAL"] = "1"
        return e

    PRE = re.compile(r"PREFILL ([\d.]+) ms over (\d+) tokens, ([\d.]+) tok/s")
    DEC = re.compile(r"DECODE ([\d.]+) ms a token")

    def parse(self, rc, out, err):
        p, d = self.PRE.search(out), self.DEC.search(out)
        t = re.search(r"TOKENS (.*)", out)
        if rc != 0 or not p or not d or not t:
            return None
        ms = float(d.group(1))
        return dict(prefill=float(p.group(3)), decode=(1000.0 / ms) if ms > 0 else 0.0, tokens_hash=str(zlib.crc32(t.group(1).encode())))


# ---------------------------------------------------------------- the manifest
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def synthetic_manifest(build, pack, fleet_path):
    """Two loopback servers (experts 0-15 / 16-31 of every layer) and the conductor here: the machinery on a toy."""
    layers = "0-1"
    srv = lambda name, port, ex: dict(host=name, bin=os.path.join(build, "strata-dsv41-expert-server"), pack=pack, listen="127.0.0.1:%d" % port, serve_layers=layers, serve_experts=ex, cpu_threads=2)
    plan = dict(remote_experts="127.0.0.1:17971=%s/0-15,127.0.0.1:17972=%s/16-31" % (layers, layers), servers=[srv("synth_a", 17971, "0-15"), srv("synth_b", 17972, "16-31")])
    man = dict(engine=dict(host="conductor", bin="ds4_net_test", pack=pack), vocab=64,
               hosts={"conductor": {}, "synth_a": {}, "synth_b": {}}, plans=dict(before=plan, after=plan),
               prompts={}, ready_timeout_s=60, runs=3, decode_tokens=8, sizes={"2k": 2048, "16k": 4096})
    fleet = [dict(name="conductor"), dict(name="synth_a"), dict(name="synth_b")]
    with open(fleet_path, "w") as f:
        json.dump(fleet, f)
    return man


def median(v):
    return statistics.median(v) if v else None


def fmt(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="the run's site paths (tools/dsv41/bbxl_run_manifest.example.json)")
    ap.add_argument("--fleet", default=os.path.join(HERE, "fleet.json"), help="the host list (BBXL S6 fleet.json)")
    ap.add_argument("--out", help="output folder (default runs/bbxl-<timestamp>)")
    ap.add_argument("--dry-run", action="store_true", help="print every command, run none")
    ap.add_argument("--arms", default="all-arms", help="comma list of arm ids (default: every arm); base is always run")
    ap.add_argument("--runs", type=int, help="timed runs per arm and prompt size (default: the manifest's, else 3)")
    ap.add_argument("--sizes", default=None, help="prompt sizes, e.g. 2k,16k (default: both)")
    ap.add_argument("--synthetic", nargs=2, metavar=("BUILD_DIR", "PACK"), help="run the machinery on a synthetic pack (ds4_net_test + loopback servers)")
    a = ap.parse_args()

    out = a.out or os.path.join("runs", "bbxl-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bbxl-")
    fleet_path = a.fleet
    if a.synthetic:
        man = synthetic_manifest(a.synthetic[0], a.synthetic[1], os.path.join(tmp, "fleet.json"))
        fleet_path = os.path.join(tmp, "fleet.json")
        engine = SyntheticEngine(man, a.synthetic[0])
        a.dry_run = a.dry_run
    else:
        if not a.manifest:
            ap.error("--manifest is required (or --synthetic BUILD_DIR PACK)")
        man = load_json(a.manifest)
        engine = RealEngine(man)
    if not os.path.exists(fleet_path):
        print("no fleet.json at %s: it comes with BBXL S6 (max2/s6-nvme-tier); pass --fleet FILE" % fleet_path, file=sys.stderr)
        return 2
    fleet = load_json(fleet_path)
    hosts = []
    for f in fleet:
        h = dict(f)
        h.update(man.get("hosts", {}).get(f["name"], {}))
        h["name"] = f["name"]
        hosts.append(h)
    by_name = {h["name"]: h for h in hosts}
    conductor = by_name.get(man["engine"].get("host", "conductor")) or by_name.get("gaming") or dict(name="conductor")
    floors = dict(FLOORS)
    for k, v in man.get("floors", {}).items():
        floors[k] = tuple(v)
    runs_n = a.runs or man.get("runs", 3)
    sizes = {k: v for k, v in man.get("sizes", SIZES).items() if not a.sizes or k in a.sizes.split(",")}
    n_decode = man.get("decode_tokens", 128)
    want = None if a.arms == "all-arms" else set(a.arms.split(",")) | {"base"}
    arms = [x for x in ARMS if want is None or x["id"] in want]
    if a.synthetic:
        skipped = [x["id"] for x in arms if not engine.supported(x)]
        arms = [x for x in arms if engine.supported(x)]
    else:
        skipped = []
        for x in arms[:]:
            if x["plan"] == "after" and "after" not in man.get("plans", {}):
                skipped.append(x["id"])
                arms.remove(x)
            if x["server"] == "gpu" and not any(s.get("gpu_bin") for p in man["plans"].values() for s in p.get("servers", [])):
                if x in arms:
                    skipped.append(x["id"])
                    arms.remove(x)
    run = Runner(a.dry_run, out)
    started = datetime.datetime.now().isoformat(timespec="seconds")

    # ---- preflight
    used = {conductor["name"]}
    for p in man["plans"].values():
        used |= {s["host"] for s in p.get("servers", [])}
    print("== preflight")
    rows, problems = preflight(run, hosts, floors, used, a.dry_run)
    for r in rows:
        print("  %-8s %-34s floor %-7s have %-8s %s" % (r[0], r[1], r[2], r[3], "ok" if r[4] else "BELOW"))
    if problems:
        print("REFUSED (nothing was started):\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3

    # ---- prompts
    base_ids = {}
    for k in sizes:
        pf = man.get("prompts", {}).get(k)
        if pf and not a.dry_run:
            base_ids[k] = [int(x) for x in open(pf, encoding="utf-8").read().replace("\n", ",").split(",") if x.strip()]
        else:
            base_ids[k] = [(i * 31 + 7) % man.get("vocab", 100000) for i in range(997)]   # (synthetic / dry run: a fixed filler)
            if not pf and not a.synthetic:
                print("note: no prompt file for %s in the manifest; a filler id list is used (the speed does not depend on the words, the routing does)" % k, file=sys.stderr)

    results = {}   # (arm, size) -> dict(prefill=[...], decode=[...], hashes=[...])
    groups = []
    for x in arms:
        key = (x["server"], x["plan"])
        for g in groups:
            if g[0] == key:
                g[1].append(x)
                break
        else:
            groups.append((key, [x]))
    code = 0
    try:
        for (kind, planname), garms in groups:
            plan = man["plans"][planname]
            print("== servers: %s, plan %s (arms: %s)" % (kind, planname, ", ".join(x["id"] for x in garms)))
            handles = []
            for s in plan.get("servers", []):
                h = by_name.get(s["host"], {"name": s["host"]})
                if s.get("managed", True) is False:
                    print("  %s: not managed by this script; checking that %s answers" % (s["host"], s["listen"]))
                else:
                    if kind == "gpu" and s.get("gpu_bin"):   # (a host with no GPU server in the manifest keeps its CPU server: the fleet is mixed)
                        argv = [s["gpu_bin"], "--pack", s["pack"], "--listen", s["listen"], "--expert-server-gpu", "--serve-layers", s["serve_layers"]]
                        if s.get("gpu_vram_gb"):
                            argv += ["--gpu-vram-gb", str(s["gpu_vram_gb"])]
                        env = {"CUDA_VISIBLE_DEVICES": s["gpu_uuid"]} if s.get("gpu_uuid") else {}
                    else:
                        argv = [s["bin"], "--pack", s["pack"], "--listen", s["listen"], "--serve-layers", s["serve_layers"], "--cpu-threads", str(s.get("cpu_threads", 4))]
                        env = {}
                        if a.synthetic:
                            argv.append("--no-warm")
                    if s.get("serve_experts"):
                        argv += ["--serve-experts", s["serve_experts"]]
                    if a.synthetic:
                        env["STRATA_EXPERT_TOKEN"] = "bbxl-synthetic"
                    handles.append(run.spawn(h, argv, env, log="server-%s-%s-%s.log" % (s["host"], kind, planname), token_file=s.get("token_file")))
                port_addr = s["listen"].replace("0.0.0.0", s.get("connect_host", h.get("addr", "127.0.0.1").split(":")[0] if h.get("addr") else "127.0.0.1"))
                if not wait_port(port_addr, man.get("ready_timeout_s", 900), a.dry_run):
                    raise Fail("server %s did not answer on %s" % (s["host"], port_addr))
            for arm in garms:
                env = engine.env(arm, plan)
                # a warm-up run (not counted): the first run of an arm pays its buffers and the page cache
                wsz = min(sizes, key=lambda k: sizes[k])
                wf = os.path.join(tmp, "warm-%s.ids" % arm["id"])
                engine.write_ids(base_ids[wsz], sizes[wsz], 9000 + len(arm["id"]), wf) if not a.dry_run else open(wf, "w").write("0")
                print("== %s warm-up (%s)" % (arm["id"], arm["what"]))
                rc, o, e = run.run(conductor, engine.argv(arm, plan, wf, n_decode), env, log="%s-warmup.log" % arm["id"])
                if not a.dry_run and engine.parse(rc, o, e) is None:
                    raise Fail("arm %s: the warm-up run failed (see %s-warmup.log)" % (arm["id"], arm["id"]))
                for sz, n in sizes.items():
                    for i in range(1, runs_n + 1):
                        idf = os.path.join(tmp, "p-%s-%s-s%d.ids" % (arm["id"], sz, i))
                        if not a.dry_run:
                            engine.write_ids(base_ids[sz], n, i, idf)
                        else:
                            open(idf, "w").write("0")
                        rc, o, e = run.run(conductor, engine.argv(arm, plan, idf, n_decode), env, log="%s-%s-run%d.log" % (arm["id"], sz, i))
                        r = None if a.dry_run else engine.parse(rc, o, e)
                        if not a.dry_run and r is None:
                            raise Fail("arm %s %s run %d failed (see %s-%s-run%d.log)" % (arm["id"], sz, i, arm["id"], sz, i))
                        d = results.setdefault((arm["id"], sz), dict(prefill=[], decode=[], hashes=[]))
                        if r:
                            d["prefill"].append(r["prefill"]); d["decode"].append(r["decode"]); d["hashes"].append(r["tokens_hash"])
            for h in handles:
                run.stop(h)
    except Fail as e:
        print("FAILED: %s" % e, file=sys.stderr)
        code = 1
    finally:
        run.stop_all()

    if a.dry_run:
        print("[dry-run] nothing was run; %d arms x %d sizes x %d runs (+ a warm-up per arm) would be" % (len(arms), len(sizes), runs_n))
        return code
    write_outputs(out, started, engine, hosts, rows, arms, skipped, sizes, runs_n, n_decode, results, man)
    print("== wrote %s and %s" % (os.path.join(out, "RESULTS.md"), os.path.join(out, "results.jsonl")))
    return code


def write_outputs(out, started, engine, hosts, pre_rows, arms, skipped, sizes, runs_n, n_decode, results, man):
    base = {sz: results.get(("base", sz)) for sz in sizes}
    L = []
    L.append("## BBXL combined run %s" % started)
    L.append("")
    L.append("Engine: %s. %d timed runs per arm and prompt (median shown, every run listed), one warm-up run per arm not counted, every run with its own 16-token unique prefix; %d decoded tokens." % (engine.name, runs_n, n_decode))
    L.append("")
    L.append("| host | check | floor | have |")
    L.append("| --- | --- | --- | --- |")
    for r in pre_rows:
        L.append("| %s | %s | %s | %s |" % (r[0], r[1], r[2], r[3]))
    L.append("")
    L.append("| stage | arm | prompt | prefill tok/s | vs base | decode tok/s | vs base | tokens = base |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for arm in arms:
        for sz in sizes:
            d = results.get((arm["id"], sz))
            if not d or not d["prefill"]:
                continue
            mp, md = median(d["prefill"]), median(d["decode"])
            b = base.get(sz)
            bp, bd = (median(b["prefill"]), median(b["decode"])) if b and b["prefill"] else (None, None)
            same = "-" if arm["id"] == "base" or not b else ("yes" if d["hashes"] == b["hashes"] else "no (rows differ in rounding or the arm changes the path)")
            L.append("| %s | %s | %s | %s (%s) | %s | %s (%s) | %s | %s |" % (
                arm["stage"], arm["id"], sz, fmt(mp), " ".join(fmt(x) for x in d["prefill"]), ("x%.2f" % (mp / bp)) if bp else "-",
                fmt(md, 2), " ".join(fmt(x, 2) for x in d["decode"]), ("x%.2f" % (md / bd)) if bd else "-", same))
            for metric, vals, unit in (("prefill", d["prefill"], "tok/s"), ("decode", d["decode"], "tok/s")):
                rows.append(dict(schema=SCHEMA, ts=started, project="bbxl", engine=engine.name, stage=arm["stage"], arm=arm["id"], arm_what=arm["what"],
                                 prompt=sz, prompt_tokens=sizes[sz], metric=metric, value=round(median(vals), 3), unit=unit, runs=[round(v, 3) for v in vals],
                                 decode_tokens=n_decode, tokens_match_base=(d["hashes"] == b["hashes"]) if (b and arm["id"] != "base") else None))
    L.append("")
    if skipped:
        L.append("Not run: %s (needs real weights, a GPU server or the 'after' plan that this manifest does not provide)." % ", ".join(skipped))
        L.append("")
    L.append("Arms:")
    for arm in arms:
        L.append("- `%s` (%s): %s" % (arm["id"], arm["stage"], arm["what"]))
    with open(os.path.join(out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
