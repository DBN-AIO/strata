#!/usr/bin/env python3
"""Synthetic split test for DeepSeek experts over the LAN (docs/DEEPSEEK41_SPLIT.md).

Builds a tiny DeepSeek-shaped pack (make_synthetic_pack.py), runs tests/dsv41/ds4_net_test once in one process (local), then
against expert servers on localhost, and compares the 64 greedy tokens and the hash of every expert row received:
  1. one server for all layers
  2. two servers splitting the layers
  3. a partial split (the conductor computes the other layers itself)
  4. a server that dies mid-run (STRATA_DS4_SERVER_DIE_AFTER): the conductor falls back to its own lane, same output
  5. a conductor of the wrong engine family (qwen38) is refused at the handshake; so are a wrong token and another geometry
Needs the built ds4_net_test and strata-dsv41-expert-server:   synthetic_split_test.py BUILD_DIR
"""
import os
import re
import subprocess
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
build = sys.argv[1] if len(sys.argv) > 1 else "build-cpu"
exe = ".exe" if os.name == "nt" else ""
test_bin = os.path.join(build, "ds4_net_test" + exe)
srv_bin = os.path.join(build, "strata-dsv41-expert-server" + exe)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" -- " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


def run(args, env=None, timeout=300):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([test_bin] + args, capture_output=True, text=True, env=e, timeout=timeout)


def start(pack, port, layers, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.Popen([srv_bin, "--pack", pack, "--listen", "127.0.0.1:%d" % port, "--serve-layers", layers, "--no-warm", "--cpu-threads", "2"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
    time.sleep(1.0)
    return p


def stop(p):
    if p.poll() is None:
        p.terminate()
    try:
        out, _ = p.communicate(timeout=10)
    except Exception:
        p.kill()
        out, _ = p.communicate()
    return out


def lines(r):
    d = {}
    for ln in r.stdout.splitlines():
        m = re.match(r"(TOKENS|ROWHASH|ROWS|STATS|OPEN_FAILED)\s*(.*)", ln)
        if m:
            d[m.group(1)] = m.group(2)
    return d


with tempfile.TemporaryDirectory() as tmp:
    pack = os.path.join(tmp, "pack")
    pack3 = os.path.join(tmp, "pack3")
    subprocess.check_call([sys.executable, os.path.join(here, "make_synthetic_pack.py"), pack, "--seed", "1"], stdout=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, os.path.join(here, "make_synthetic_pack.py"), pack3, "--experts", "16"], stdout=subprocess.DEVNULL)
    ref = run(["--pack", pack, "--mode", "local", "--tokens", "64"])
    R = lines(ref)
    check("reference run", ref.returncode == 0 and "TOKENS" in R, ref.stderr[-300:])
    check("reference is finite and not degenerate", "FINITE 1" in R.get("ROWS", "") and len(set(R.get("TOKENS", "").split())) >= 8, R.get("TOKENS", ""))
    token = {"STRATA_EXPERT_TOKEN": "synthetic-test-token"}
    # 1. one server, every layer
    s1 = start(pack, 17840, "0-7", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17840=0-7"], token)
    stop(s1)
    L = lines(r)
    check("one server: 64 tokens identical", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS"), r.stderr[-300:] + r.stdout[-200:])
    check("one server: every expert row bit-identical", L.get("ROWHASH") == R.get("ROWHASH"))
    check("one server: no row was computed locally", re.search(r"remote expert rows [1-9]\d*, computed locally 0,", L.get("STATS", "")) is not None, L.get("STATS", ""))
    # 2. two servers, the layers split
    a = start(pack, 17841, "0-3", token)
    b = start(pack, 17842, "4-7", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17841=0-3,127.0.0.1:17842=4-7"], token)
    stop(a)
    stop(b)
    L = lines(r)
    check("two servers (layers 0-3 / 4-7): tokens and rows identical", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-300:])
    # 3. only some layers handed out
    c = start(pack, 17843, "2-5", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17843=2-5"], token)
    stop(c)
    L = lines(r)
    check("partial split (layers 2-5 remote): identical", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"))
    # 4. the server dies mid-run
    d = start(pack, 17844, "0-7", dict(token, STRATA_DS4_SERVER_DIE_AFTER="40"))
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17844=0-7"], dict(token, STRATA_EXPERT_TIMEOUT_MS="1500"))
    stop(d)
    L = lines(r)
    m = re.search(r"remote expert rows (\d+), computed locally (\d+)", L.get("STATS", ""))
    check("server dies after 40 requests: output identical via the local fallback", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-300:])
    check("server dies: both remote and local rows were used", m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, L.get("STATS", ""))
    # 5. refusals
    e = start(pack, 17845, "0-7", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--arch", "0", "--expect-open-fail"], token)
    check("a qwen38 conductor is refused", "engine family differs" in r.stdout, r.stdout + r.stderr[-200:])
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--expect-open-fail"], {"STRATA_EXPERT_TOKEN": "wrong"})
    check("a wrong token is refused", "authentication failed" in r.stdout, r.stdout)
    r = run(["--pack", pack3, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--expect-open-fail"], token)
    check("another expert count is refused", "OPEN_FAILED" in r.stdout, r.stdout)
    stop(e)

print("%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
