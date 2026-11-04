#!/usr/bin/env python3
"""Synthetic split test for DeepSeek experts over the LAN (docs/DEEPSEEK41_SPLIT.md).

Builds a tiny DeepSeek-shaped pack (make_synthetic_pack.py), runs tests/dsv41/ds4_net_test once in one process (local), then
against expert servers on localhost, and compares the 64 greedy tokens and the hash of every expert row received:
  1. one server for all layers
  2. two servers splitting the layers
  3. a partial split (the conductor computes the other layers itself)
  3c-3e. (BBXL S4) one layer's experts on two servers asked at the same time; the conductor's own experts computed while the servers
         answer (Remote::begin / finish); both with an injected delay (server: STRATA_DS4_SERVER_DELAY_MS, conductor: --local-delay-ms)
         so the timing assertions are deterministic on a loaded host
  4. a server that dies mid-run (STRATA_DS4_SERVER_DIE_AFTER): the conductor falls back to its own lane, same output
  4c. (BBXL S3) experts renumbered by popularity: the conductor calls ranks (STRATA_DS4_EXPERT_PERM), the slice is laid out by rank
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


def start(pack, port, layers, env=None, experts=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.Popen([srv_bin, "--pack", pack, "--listen", "127.0.0.1:%d" % port, "--serve-layers", layers, "--no-warm", "--cpu-threads", "2"]
                         + (["--serve-experts", experts] if experts else []),
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
        m = re.match(r"(TOKENS|ROWHASH|ROWS|STATS|OPEN_FAILED|DECODE)\s*(.*)", ln)
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
    # 3b. every layer's experts split over two hosts (B5): experts 0-15 here, 16-31 there, asked at the same time
    x1 = start(pack, 17849, "0-7", token, experts="0-15")
    x2 = start(pack, 17850, "0-7", token, experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--remote", "127.0.0.1:17849=0-7/0-15,127.0.0.1:17850=0-7/16-31"], token)
    stop(x1)
    stop(x2)
    L = lines(r)
    check("experts split over two hosts (0-15 / 16-31): identical, none local", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH")
          and "computed locally 0," in L.get("STATS", ""), r.stderr[-300:] + L.get("STATS", ""))
    # 3c. BBXL S4: every layer's experts on TWO servers, each answering in 30 ms (injected).  Asked one after the other a layer would cost
    # 60 ms (480 ms a token over 8 layers); asked at the same time, 30 ms (240 ms).  Same tokens and rows as the local run.
    DEL = 30
    ref6 = run(["--pack", pack, "--mode", "local", "--tokens", "6"])
    R6 = lines(ref6)
    dl = dict(token, STRATA_DS4_SERVER_DELAY_MS=str(DEL))
    y1 = start(pack, 17853, "0-7", dl, experts="0-15")
    y2 = start(pack, 17854, "0-7", dl, experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--tokens", "6", "--time", "--remote", "127.0.0.1:17853=0-7/0-15,127.0.0.1:17854=0-7/16-31"], token)
    rs = run(["--pack", pack, "--mode", "remote", "--tokens", "6", "--time", "--remote", "127.0.0.1:17853=0-7/0-15,127.0.0.1:17854=0-7/16-31"], dict(token, STRATA_NET_SERIAL="1"))
    stop(y1)
    stop(y2)
    LSr = lines(rs)
    msr = re.match(r"([\d.]+) ms a token", LSr.get("DECODE", ""))
    msr = float(msr.group(1)) if msr else -1.0
    check("the serial arm (STRATA_NET_SERIAL=1, the measurement hook): same tokens and rows", rs.returncode == 0 and LSr.get("TOKENS") == R6.get("TOKENS") and LSr.get("ROWHASH") == R6.get("ROWHASH"), rs.stderr[-300:])
    check("the serial arm costs the sum of the servers (8 x 2 x %d ms): %.0f ms" % (DEL, msr), msr >= 0.9 * 16 * DEL, "%.1f ms" % msr)
    L = lines(r)
    m = re.match(r"([\d.]+) ms a token", L.get("DECODE", ""))
    ms = float(m.group(1)) if m else -1.0
    check("two delayed servers: tokens and rows identical", r.returncode == 0 and L.get("TOKENS") == R6.get("TOKENS") and L.get("ROWHASH") == R6.get("ROWHASH"), r.stderr[-300:] + L.get("STATS", ""))
    check("two delayed servers: a token costs ~ one server's time (8 x %d ms), not the sum (%d ms): %.0f ms" % (DEL, 16 * DEL, ms),
          0.9 * 8 * DEL <= ms <= 1.5 * 8 * DEL, "%.1f ms" % ms)
    # 3d. BBXL S4: the conductor's own experts (0-15) cost 25 ms a layer (injected) while the one server (16-31) takes 30 ms.  One after the other:
    # 55 ms a layer (440 ms a token); overlapped (Remote::begin, the local work, Remote::finish): about 30 ms (240 ms).
    LOC = 25
    z = start(pack, 17855, "0-7", dl, experts="16-31")
    seqr = run(["--pack", pack, "--mode", "remote", "--tokens", "6", "--time", "--local-delay-ms", str(LOC), "--remote", "127.0.0.1:17855=0-7/16-31"], token)
    ovr = run(["--pack", pack, "--mode", "remote", "--tokens", "6", "--time", "--overlap", "--local-delay-ms", str(LOC), "--remote", "127.0.0.1:17855=0-7/16-31"], token)
    stop(z)
    LS, LO = lines(seqr), lines(ovr)
    def msof(d):
        mm = re.match(r"([\d.]+) ms a token", d.get("DECODE", ""))
        return float(mm.group(1)) if mm else -1.0
    ms_s, ms_o = msof(LS), msof(LO)
    check("conductor work, one after the other: tokens and rows identical", seqr.returncode == 0 and LS.get("TOKENS") == R6.get("TOKENS") and LS.get("ROWHASH") == R6.get("ROWHASH"), seqr.stderr[-300:])
    check("conductor work overlapped with the server: tokens and rows identical", ovr.returncode == 0 and LO.get("TOKENS") == R6.get("TOKENS") and LO.get("ROWHASH") == R6.get("ROWHASH"), ovr.stderr[-300:] + ovr.stdout[-200:])
    check("one after the other costs the sum (8 x 55 ms): %.0f ms" % ms_s, ms_s >= 0.9 * 8 * (DEL + LOC), "%.1f ms" % ms_s)
    check("overlapped costs the longer of the two (8 x 30 ms), not the sum: %.0f ms" % ms_o, 0.9 * 8 * DEL <= ms_o <= 0.8 * 8 * (DEL + LOC), "%.1f ms" % ms_o)
    # 3e. the overlapped path with a server that dies mid-run: its pending rows come back to the conductor's own lane, same output
    z = start(pack, 17856, "0-7", dict(token, STRATA_DS4_SERVER_DIE_AFTER="40"), experts="16-31")
    r = run(["--pack", pack, "--mode", "remote", "--overlap", "--remote", "127.0.0.1:17856=0-7/16-31"], dict(token, STRATA_EXPERT_TIMEOUT_MS="1500"))
    stop(z)
    L = lines(r)
    m = re.search(r"remote expert rows (\d+), computed locally (\d+)", L.get("STATS", ""))
    check("overlapped, server dies after 40 requests: identical via the local fallback", r.returncode == 0 and L.get("TOKENS") == R.get("TOKENS") and L.get("ROWHASH") == R.get("ROWHASH"), r.stderr[-300:] + r.stdout[-200:])
    check("overlapped, server dies: remote and local rows both used", m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, L.get("STATS", ""))
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

print("%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
