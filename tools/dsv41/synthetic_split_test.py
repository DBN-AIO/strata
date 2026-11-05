#!/usr/bin/env python3
"""Synthetic split test for DeepSeek experts over the LAN (docs/DEEPSEEK41_SPLIT.md).

Builds a tiny DeepSeek-shaped pack (make_synthetic_pack.py), runs tests/dsv41/ds4_net_test once in one process (local), then
against expert servers on localhost, and compares the 64 greedy tokens and the hash of every expert row received:
  1. one server for all layers
  2. two servers splitting the layers
  3. a partial split (the conductor computes the other layers itself)
  4. a server that dies mid-run (STRATA_DS4_SERVER_DIE_AFTER): the conductor falls back to its own lane, same output
  4c. (BBXL S3) experts renumbered by popularity: the conductor calls ranks (STRATA_DS4_EXPERT_PERM), the slice is laid out by rank
  5. a conductor of the wrong engine family (qwen38) is refused at the handshake; so are a wrong token and another geometry
  6. speculative decoding (the DSpark drafter's shape): drafts verified as ONE window per layer through the servers; the tokens equal plain
     greedy decoding's, drafts are accepted (count > 0), and the servers see one call per layer per window, not one per draft token
Needs the built ds4_net_test and strata-dsv41-expert-server:   synthetic_split_test.py [BUILD_DIR] [--tmp DIR]

Scratch space: the packs (about 45 MB apparent at the peak, measured) never go to the system temp dir (/tmp is a shared, quota-limited
tmpfs on the CI runner and was full of other jobs' leftovers: "Disk quota exceeded", CI runs #13-#16).  `--tmp DIR` (else $RUNNER_TEMP, else BUILD_DIR) is the parent; the test makes its own
subdirectory there, points TMPDIR of every child at it and removes it on exit, a failure or SIGTERM included.  It refuses to start when that
directory has less than NEED_GIB free (default 1, STRATA_TEST_NEED_GIB) or is a tmpfs/ramfs: a clear message instead of a mid-run write error.
Servers are waited for by polling their port (30 s), not a fixed sleep; STRATA_TEST_SERVER_START_DELAY=SECONDS delays each server's start to
prove it.
"""
import argparse
import atexit
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
ap_ = argparse.ArgumentParser()
ap_.add_argument("build", nargs="?", default="build-cpu")
ap_.add_argument("--tmp", default=None, help="parent directory for the scratch packs (default $RUNNER_TEMP, else the build dir; never /tmp)")
args_ = ap_.parse_args()
build = args_.build
exe = ".exe" if os.name == "nt" else ""
test_bin = os.path.join(build, "ds4_net_test" + exe)
srv_bin = os.path.join(build, "strata-dsv41-expert-server" + exe)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" -- " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


NEED_GIB = float(os.environ.get("STRATA_TEST_NEED_GIB", "1"))


def fs_type(path):
    """Filesystem type of the mount holding path (Linux /proc/mounts; "" when unknown)."""
    best, typ = "", ""
    try:
        real = os.path.realpath(path)
        for ln in open("/proc/mounts"):
            f = ln.split()
            mp = f[1].replace("\\040", " ")
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best):
                best, typ = mp, f[2]
    except OSError:
        pass
    return typ


def scratch_parent():
    base = args_.tmp or os.environ.get("RUNNER_TEMP") or build
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)
    free = shutil.disk_usage(base).free / 2 ** 30
    typ = fs_type(base)
    if typ in ("tmpfs", "ramfs"):
        sys.exit("synthetic_split_test: scratch dir %s is on %s (RAM-backed, size/quota limited; the packs need up to %g GiB): "
                 "pass --tmp DIR on a disk, e.g. under the workspace" % (base, typ, NEED_GIB))
    if free < NEED_GIB:
        sys.exit("synthetic_split_test: scratch dir %s has %.1f GiB free, the packs need %g GiB: pass --tmp DIR on a bigger disk" % (base, free, NEED_GIB))
    return base


def run(args, env=None, timeout=300):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([test_bin] + args, capture_output=True, text=True, env=e, timeout=timeout)


_servers = []


def _kill_servers():   # a failure or SIGTERM must not leave servers running on the pack that is being removed
    for q in _servers:
        if q.poll() is None:
            q.kill()


atexit.register(_kill_servers)


def start(pack, port, layers, env=None, experts=None):
    e = dict(os.environ)
    e.update(env or {})
    cmd = [srv_bin]
    delay = os.environ.get("STRATA_TEST_SERVER_START_DELAY")   # fault injection: the server binds late, as on a loaded runner
    if delay:
        cmd = [sys.executable, "-c", "import os,sys,time;time.sleep(float(sys.argv[1]));os.execv(sys.argv[2],sys.argv[2:])", delay, srv_bin]
    p = subprocess.Popen(cmd + ["--pack", pack, "--listen", "127.0.0.1:%d" % port, "--serve-layers", layers, "--no-warm", "--cpu-threads", "2"]
                         + (["--serve-experts", experts] if experts else []),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
    _servers.append(p)
    check("server on :%d is listening" % port, wait_port(p, port), "not listening after 30 s, or exited early")
    return p


def wait_port(p, port, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if p.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


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
        m = re.match(r"(TOKENS|ROWHASH|ROWS|STATS|OPEN_FAILED|SPEC|DECODE)\s*(.*)", ln)
        if m:
            d[m.group(1)] = m.group(2)
    return d


def _term(sig, frm):
    raise SystemExit(128 + sig)   # unwinds the with-block below so the scratch dir is removed


signal.signal(signal.SIGTERM, _term)
parent = scratch_parent()
with tempfile.TemporaryDirectory(prefix="synth-split-", dir=parent) as tmp:
    os.environ["TMPDIR"] = tmp    # every child (make_synthetic_pack, the servers) stays under the same removable directory
    tempfile.tempdir = None
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
    # 3b. every layer's experts split over two hosts (B5): experts 0-15 here, 16-31 there, asked at the same time
    x1 = start(pack, 17849, "0-7", token, experts="0-15")
    x2 = start(pack, 17850, "0-7", token, experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17849=0-7/0-15,127.0.0.1:17850=0-7/16-31"], token)
    stop(x1)
    stop(x2)
    L = lines(r)
    check("experts split over two hosts (0-15 / 16-31): identical, none local", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH")
          and "computed locally 0," in L.get("STATS", ""), r.stderr[-300:] + L.get("STATS", ""))
    # 4. the server dies mid-run
    d = start(pack, 17844, "0-7", dict(token, STRATA_DS4_SERVER_DIE_AFTER="40"))
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17844=0-7"], dict(token, STRATA_EXPERT_TIMEOUT_MS="1500"))
    stop(d)
    L = lines(r)
    m = re.search(r"remote expert rows (\d+), computed locally (\d+)", L.get("STATS", ""))
    check("server dies after 40 requests: output identical via the local fallback", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-300:])
    check("server dies: both remote and local rows were used", m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, L.get("STATS", ""))
    # 4b. a server on a sparse slice (only its layers' bytes on disk): slice_stream.py emit | apply
    sl = os.path.join(tmp, "slice")
    em = subprocess.Popen([sys.executable, os.path.join(here, "slice_stream.py"), "emit", pack, "2-5"], stdout=subprocess.PIPE)
    ap = subprocess.run([sys.executable, os.path.join(here, "slice_stream.py"), "apply", sl], stdin=em.stdout, capture_output=True, text=True)
    em.wait()
    check("slice_stream emit | apply", em.returncode == 0 and ap.returncode == 0, ap.stderr[-200:])
    h = start(sl, 17848, "2-5", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17848=2-5"], token)
    stop(h)
    L = lines(r)
    check("a server on a sparse slice of layers 2-5: identical", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-200:])
    # 4c. BBXL S3: the experts renumbered by popularity.  The conductor translates every remote call to the expert's RANK
    # (STRATA_DS4_EXPERT_PERM); the slice holds the original expert's bytes at its rank's slot (slice_stream.py emit --perm).
    # Ranks 0-15 stay on the conductor (its hot ones), ranks 16-31 are the server's; same tokens and expert rows as the local run.
    import random
    import struct
    rng = random.Random(11)
    perm = []
    for _ in range(8):
        pr = list(range(32))
        rng.shuffle(pr)
        perm.append(pr)
    pf = os.path.join(tmp, "perm.bin")
    with open(pf, "wb") as fh:
        fh.write(b"DS4PERM1" + struct.pack("<II", 8, 32))
        for pr in perm:
            fh.write(struct.pack("<32H", *pr))
    ps = os.path.join(tmp, "pslice")
    em = subprocess.Popen([sys.executable, os.path.join(here, "slice_stream.py"), "emit", pack, "0-7", "16-31", "--perm", pf], stdout=subprocess.PIPE)
    ap = subprocess.run([sys.executable, os.path.join(here, "slice_stream.py"), "apply", ps], stdin=em.stdout, capture_output=True, text=True)
    em.wait()
    check("slice_stream emit --perm | apply", em.returncode == 0 and ap.returncode == 0, ap.stderr[-200:])
    pp = start(ps, 17851, "0-7", token, experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17851=0-7/16-31"], dict(token, STRATA_DS4_EXPERT_PERM=pf))
    stop(pp)
    L = lines(r)
    m = re.search(r"remote expert rows (\d+), computed locally (\d+)", L.get("STATS", ""))
    check("renumbered experts (ranks 16-31 remote, 0-15 the conductor's): tokens and rows identical",
          r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-300:] + L.get("STATS", ""))
    check("renumbered experts: some rows remote, some local", m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, L.get("STATS", ""))
    bad = os.path.join(tmp, "badperm.bin")
    with open(bad, "wb") as fh:
        fh.write(b"DS4PERM1" + struct.pack("<II", 8, 32) + struct.pack("<32H", *([0] * 32)) * 8)
    pp = start(ps, 17852, "0-7", token, experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17852=0-7/16-31", "--expect-open-fail"], dict(token, STRATA_DS4_EXPERT_PERM=bad))
    check("a perm file that is not a permutation is refused", "STRATA_DS4_EXPERT_PERM" in r.stdout, r.stdout + r.stderr[-200:])
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17852=0-7/16-31", "--expect-open-fail"], dict(token, STRATA_DS4_EXPERT_PERM=os.path.join(tmp, "nope.bin")))
    check("a missing perm file is refused", "STRATA_DS4_EXPERT_PERM" in r.stdout, r.stdout + r.stderr[-200:])
    stop(pp)
    # 5. refusals
    e = start(pack, 17845, "0-7", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--arch", "0", "--expect-open-fail"], token)
    check("a qwen38 conductor is refused", "engine family differs" in r.stdout, r.stdout + r.stderr[-200:])
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--expect-open-fail"], {"STRATA_EXPERT_TOKEN": "wrong"})
    check("a wrong token is refused", "authentication failed" in r.stdout, r.stdout)
    r = run(["--pack", pack3, "--mode", "remote", "--remote", "127.0.0.1:17845=0-7", "--expect-open-fail"], token)
    check("another expert count is refused", "OPEN_FAILED" in r.stdout, r.stdout)
    stop(e)

    # 6. speculative decoding through the servers (a drafter, a verify window of k+1 tokens, the longest matching prefix kept)
    plain = run(["--pack", pack, "--mode", "local", "--causal", "--tokens", "64", "--prompt", "40", "--prompt-seed", "5"])
    P = lines(plain)
    check("causal reference run (plain greedy)", plain.returncode == 0 and "TOKENS" in P, plain.stderr[-300:])
    sp = lines(run(["--pack", pack, "--mode", "local", "--causal", "--tokens", "64", "--prompt", "40", "--prompt-seed", "5", "--spec", "4", "--draft-acc", "0.7"]))
    check("spec local (oracle drafter, 70% per draft): tokens equal plain greedy", sp.get("TOKENS") == P.get("TOKENS") and "TOKENS" in sp, sp.get("TOKENS", "")[:80])

    def spec_nums(d):
        m = re.search(r"k (\d+) rounds (\d+) drafted (\d+) accepted (\d+) windows (\d+) layer_calls (\d+) wire_calls (\d+)", d.get("SPEC", ""))
        return [int(x) for x in m.groups()] if m else None
    sn = spec_nums(sp)
    check("spec local: the drafter was used (accepted > 0)", sn is not None and sn[3] > 0, sp.get("SPEC", ""))
    sg = lines(run(["--pack", pack, "--mode", "local", "--causal", "--tokens", "64", "--prompt", "40", "--prompt-seed", "5", "--spec", "4"]))
    check("spec local (n-gram drafter): tokens equal plain greedy", sg.get("TOKENS") == P.get("TOKENS") and "TOKENS" in sg)
    f = start(pack, 17851, "0-7", token)
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17851=0-7", "--causal", "--tokens", "64", "--prompt", "40", "--prompt-seed", "5",
             "--spec", "4", "--draft-acc", "0.7"], token)
    stop(f)
    SR = lines(r)
    check("spec through one server: tokens equal plain greedy and the local spec run", r.returncode == 0 and SR.get("TOKENS") == P.get("TOKENS"), r.stderr[-300:])
    check("spec through one server: every expert row bit-identical to the local spec run", "ROWHASH" in SR and SR.get("ROWHASH") == sp.get("ROWHASH"))
    sr = spec_nums(SR)
    check("spec through one server: drafts accepted", sr is not None and sr[3] > 0, SR.get("SPEC", ""))
    # the verify window is one wire call per layer: wire calls == layer calls (prompt chunks + windows), none one-per-draft-token
    check("spec through one server: one call per layer per window (wire calls == layer calls)", sr is not None and sr[5] == sr[6] and sr[6] > 0, SR.get("SPEC", ""))
    # and speculating beats a call per token: plain decode makes n_new - 1 windows, spec makes fewer
    pn = spec_nums(lines(run(["--pack", pack, "--mode", "local", "--causal", "--tokens", "64", "--prompt", "40", "--prompt-seed", "5", "--spec", "1", "--draft-acc", "0"])))
    check("spec: fewer layer calls than one window per token", sr is not None and pn is not None and sr[5] < pn[5], "%s vs %s" % (sr, pn))

print("%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
