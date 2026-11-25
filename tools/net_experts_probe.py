#!/usr/bin/env python3
"""Black-box checks of a running `strata --expert-server` (docs/REMOTE_EXPERTS.md, "Security").

Speaks the SXP1 v2 wire protocol with the standard library only.  Needs a server that is idle (it holds one session at a
time) and, for the token checks, started with STRATA_EXPERT_TOKEN set (give the same value in this script's environment).

  STRATA_EXPERT_TOKEN=... python3 tools/net_experts_probe.py --host 127.0.0.1 --port 7840 [--timeout-ms 3000]

--timeout-ms must be the server's --expert-timeout-ms (default 10000): the idle-connection check waits that long.
For the DeepSeek server (strata-dsv41-expert-server) on a synthetic pack add --pack DIR: the model's geometry and the engine family
are read from DIR/ds4.txt (the qwen38 engine's are the defaults).  CI (.github/workflows/build-linux.yml) runs it that way, and
passes --idle-ms with the server's --expert-idle-timeout-ms (check 10 is skipped without it).
Checks (each prints PASS / FAIL):
  1. a wrong token is refused ("authentication failed"), as is no token when the server wants one
  2. an oversized pre-auth frame is dropped without the server allocating it
  3. a good HELLO (fingerprint read back from the server's mismatch answer) opens the session
  4. every kind of bad request (id out of range / negative, wrong k, window too big, bad arithmetic, bad mask, layer outside
     the served range, truncated payload) is answered with ERR and the session KEEPS serving
  5. a good request after the bad ones returns the right rows, and the same request twice returns the same bytes
  6. a second conductor is told "busy" while the first holds the session
  7. an idle connection (no HELLO) does not stop a real conductor, and is dropped after the timeout
  8. more unauthenticated connections from ONE address than the per-source cap (2) are closed at once, a conductor from another
     address still connects, and the address gets its slots back when its connections end (needs two loopback addresses: 127.0.0.2)
  9. a session that sends one byte of a request header and then nothing is dropped after the timeout
 10. a session that sends nothing at all is dropped after the idle limit (--idle-ms; skipped without it)
Exit status 0 when every check passed.
"""
import argparse, hashlib, hmac, os, random, re, socket, struct, sys, time

MAGIC = 0x31505853
HELLO, ACK, REQ, RESP, ERR, BYE, CHALLENGE = 1, 2, 3, 4, 5, 6, 7
H = 2560  # n_embd of the engine's expert kernels (set from --pack, then from the server's HELLO_ACK)
ARCH = 0  # HELLO's engine family: 0 qwen38, 1 dsv41
GEOM = (2560, 512, 48)  # n_embd, n_expert, n_layers of the qwen38 engine's model

fails = 0


def check(ok, what, extra=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + what + (("  [" + extra + "]") if extra else ""))
    if not ok:
        fails += 1


def recv_n(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b += c
    return b


def send_frame(s, typ, payload=b""):
    s.sendall(struct.pack("<IIQ", MAGIC, typ, len(payload)) + payload)


def recv_frame(s):
    magic, typ, ln = struct.unpack("<IIQ", recv_n(s, 16))
    assert magic == MAGIC, "bad magic"
    return typ, recv_n(s, ln)


def connect(a):
    s = socket.create_connection((a.host, a.port), timeout=15)
    typ, p = recv_frame(s)
    assert typ == CHALLENGE and len(p) == 40, "no challenge"
    ver, flags = struct.unpack("<II", p[:8])
    return s, ver, flags, p[8:40]


def hello(token, nonce, fp, isa, n_embd, n_expert, n_layers, version=2):
    body = struct.pack("<6I32s", version, n_embd, n_expert, n_layers, isa, ARCH, fp)
    mac = hmac.new(token, nonce + body, hashlib.sha256).digest() if token is not None else bytes(32)
    return body + mac


def try_hello(a, token, fp=bytes(32), isa=0, geom=None):
    s, ver, flags, nonce = connect(a)
    send_frame(s, HELLO, hello(token, nonce, fp, isa, *(geom or GEOM)))
    typ, p = recv_frame(s)
    return s, typ, p


def req(layer, n_tok, k, arith, ids, mask, seq=1, nx=None, extra=0):
    n = n_tok * k
    mlen = (n + 3) // 4 * 4
    x = struct.pack("<%df" % (n_tok * H if nx is None else nx), *[(i % 17) * 0.01 - 0.08 for i in range(n_tok * H if nx is None else nx)])
    return (struct.pack("<4IQ", layer, n_tok, k, arith, seq) + struct.pack("<%di" % len(ids), *ids) +
            bytes(mask) + bytes(mlen - len(mask)) + x + bytes(extra))


def read_pack(path):
    """The geometry of a synthetic pack (tools/dsv41/make_synthetic_pack.py): DIR/ds4.txt holds 'key value' lines."""
    kv = {}
    with open(os.path.join(path, "ds4.txt")) as f:
        for ln in f:
            p = ln.split(None, 1)
            if len(p) == 2:
                kv[p[0]] = p[1].strip()
    return int(kv["hidden_size"]), int(kv["n_routed_experts"]), int(kv["num_hidden_layers"])


def read_until_closed(s, limit_s):
    """Seconds until the server closes `s` (None: still open after limit_s)."""
    s.settimeout(limit_s)
    t0 = time.time()
    try:
        while True:
            if s.recv(4096) == b"":
                return time.time() - t0
    except socket.timeout:
        return None
    except ConnectionError:
        return time.time() - t0


def main():
    global H, ARCH, GEOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7840)
    ap.add_argument("--timeout-ms", type=int, default=10000)
    ap.add_argument("--pack", help="synthetic dsv41 pack directory: take the geometry and the engine family (dsv41) from it")
    ap.add_argument("--max-tok", type=int, default=0, help="the server's window limit in tokens (default 8, or 32 with --pack)")
    ap.add_argument("--idle-ms", type=int, default=0, help="the server's --expert-idle-timeout-ms: runs check 10")
    ap.add_argument("--attacker-addr", default="127.0.0.2", help="a second local address to connect from in check 8")
    a = ap.parse_args()
    if a.pack:
        GEOM = read_pack(a.pack)
        H, ARCH = GEOM[0], 1
    max_tok = a.max_tok or (32 if a.pack else 8)
    tok = os.environ.get("STRATA_EXPERT_TOKEN")
    token = tok.encode() if tok else None

    # ---- 1. authentication
    s, ver, flags, nonce = connect(a)
    s.close()
    check(ver == 2, "server speaks protocol version 2", "version %d" % ver)
    wants = bool(flags & 1)
    print("server asks for a token: %s; this script has one: %s" % (wants, token is not None))
    if wants:
        s, typ, p = try_hello(a, b"definitely-not-the-token")
        check(typ == ERR and b"authentication failed" in p, "wrong token refused", p.decode(errors="replace"))
        s.close()
        s, typ, p = try_hello(a, None)
        check(typ == ERR and b"authentication failed" in p, "missing token refused", p.decode(errors="replace"))
        s.close()
        if token is None:
            print("set STRATA_EXPERT_TOKEN to run the rest")
            return 1

    # ---- 2. pre-auth allocation bound
    s, *_ = connect(a)
    s.sendall(struct.pack("<IIQ", MAGIC, HELLO, 1 << 30))
    s.settimeout(5)
    try:
        dropped = s.recv(1) == b""
    except (ConnectionError, socket.timeout):
        dropped = True
    check(dropped, "an oversized frame before authentication is dropped")
    s.close()

    # ---- 3. read the server's fingerprint from the refusal, then open a real session
    s, typ, p = try_hello(a, token)
    m = re.search(rb"server ([0-9a-f]{64}) \(isa (\d+)\)", p)
    check(typ == ERR and m is not None, "a HELLO with the wrong model fingerprint is refused and names the server's", p[:60].decode(errors="replace"))
    s.close()
    if not m:
        return 1
    fp, isa = bytes.fromhex(m.group(1).decode()), int(m.group(2))
    s, typ, p = try_hello(a, token, fp, isa)
    check(typ == ACK and len(p) == 32, "a good HELLO opens the session")
    if typ != ACK:
        print(p.decode(errors="replace"))
        return 1
    ver, ok, lo, hi, n_embd, n_expert, top_k, sisa = struct.unpack("<8I", p)
    H = n_embd
    print("server: layers %d-%d, n_embd %d, n_expert %d, top_k %d" % (lo, hi, n_embd, n_expert, top_k))
    k, layer = top_k, lo
    good_ids = list(range(k))

    # ---- 4. bad requests: ERR each time, the session survives
    bad = [
        ("expert id == n_expert", req(layer, 1, k, 0, [n_expert] + good_ids[1:], [1] * k)),
        ("negative expert id", req(layer, 1, k, 0, [-1] + good_ids[1:], [1] * k)),
        ("huge expert id", req(layer, 1, k, 0, [2**31 - 1] + good_ids[1:], [1] * k)),
        ("k != model top-k", req(layer, 1, k + 1, 0, list(range(k + 1)), [1] * (k + 1))),
        ("window beyond the tables", req(layer, max_tok + 1, k, 0, [1] * ((max_tok + 1) * k), [1] * ((max_tok + 1) * k))),
        ("unknown arithmetic mode", req(layer, 1, k, 9, good_ids, [1] * k)),
        ("mask byte 2", req(layer, 1, k, 0, good_ids, [2] + [1] * (k - 1))),
        ("layer below the served range", req(max(lo - 1, 0) if lo > 0 else 9999, 1, k, 0, good_ids, [1] * k)),
        ("layer above the served range", req(hi + 1, 1, k, 0, good_ids, [1] * k)),
        ("payload longer than declared", req(layer, 1, k, 0, good_ids, [1] * k, extra=8)),
        ("payload truncated", req(layer, 1, k, 0, good_ids, [1] * k)[:-4]),
    ]
    alive = True
    for name, payload in bad:
        try:
            send_frame(s, REQ, payload)
            typ, p = recv_frame(s)
            check(typ == ERR, "bad request refused: " + name, p.decode(errors="replace")[:70])
        except Exception as e:
            alive = False
            check(False, "bad request refused: " + name, repr(e))
            break

    # ---- 5. the session still works
    ref = None
    for rep in range(2):
        try:
            send_frame(s, REQ, req(layer, 1, k, 0, good_ids, [1] * k, seq=100 + rep))
            typ, p = recv_frame(s)
            seq, us, rows, _ = struct.unpack("<QdII", p[:24])
            body = p[24:]
            good = typ == RESP and seq == 100 + rep and rows == k and len(body) == k * H * 4
            check(good, "good request after the bad ones returns %d rows (repeat %d)" % (k, rep + 1), "%.0f us" % us)
            if rep == 0:
                ref = body
            else:
                check(body == ref, "the same request returns the same bytes")
        except Exception as e:
            check(False, "good request after the bad ones", repr(e))
            break

    # ---- 6. a second conductor while this one holds the session
    s2, typ, p = try_hello(a, token, fp, isa)
    check(typ == ERR and b"busy" in p, "a second conductor is told busy", p.decode(errors="replace"))
    s2.close()

    # ---- 7. an idle connection must not block a real conductor
    idle, *_ = connect(a)
    send_frame(s, BYE)
    s.close()
    time.sleep(0.3)
    t0 = time.time()
    s3, typ, p = try_hello(a, token, fp, isa)
    check(typ == ACK and time.time() - t0 < a.timeout_ms / 1000.0, "a conductor connects while another connection sits idle", "%.2f s" % (time.time() - t0))
    send_frame(s3, BYE)
    s3.close()
    idle.settimeout(a.timeout_ms / 1000.0 + 5)
    t0 = time.time()
    try:
        closed = idle.recv(1) == b""
    except socket.timeout:
        closed = False
    except ConnectionError:
        closed = True
    dt = time.time() - t0
    check(closed and dt < a.timeout_ms / 1000.0 + 3, "the idle connection is dropped after the timeout", "%.1f s (timeout %d ms)" % (dt, a.timeout_ms))
    idle.close()

    # ---- 8. one address cannot take every handshake slot
    cap = 2  # netproto::kMaxHandshakesPerSource
    time.sleep(0.3)
    held = []
    try:
        for _ in range(cap + 3):
            c = socket.socket()
            c.bind((a.attacker_addr, 0))
            c.settimeout(15)
            c.connect((a.host, a.port))
            held.append(c)
    except OSError as e:
        print("SKIP  per-source handshake cap: cannot connect from %s (%s)" % (a.attacker_addr, e))
        held, cap = [], -1
    if cap > 0:
        got, dropped = 0, 0
        for c in held:
            c.settimeout(2)
            try:
                d = c.recv(16 + 40)
                if len(d) >= 16 and struct.unpack("<I", d[:4])[0] == MAGIC:
                    got += 1
                else:
                    dropped += 1
            except socket.timeout:
                dropped = dropped  # neither a challenge nor a close: counted below as neither
            except ConnectionError:
                dropped += 1
        check(got == cap and dropped == len(held) - cap, "an address holding %d unauthenticated connections is refused the rest (%d challenged, %d closed of %d)" % (cap, got, dropped, len(held)))
        t0 = time.time()
        s4, typ, p = try_hello(a, token, fp, isa)
        check(typ == ACK and time.time() - t0 < a.timeout_ms / 1000.0, "a conductor from another address connects while that address holds its slots", "%.2f s" % (time.time() - t0))
        send_frame(s4, BYE)
        s4.close()
        for c in held:
            c.close()
        time.sleep(0.5)
        t0, back = time.time(), False
        while time.time() - t0 < 3 and not back:   # the closed connections end their handshakes: the address has its slots again
            c = socket.socket()
            c.bind((a.attacker_addr, 0))
            c.settimeout(2)
            c.connect((a.host, a.port))
            try:
                back = len(c.recv(56)) == 56
            except (socket.timeout, ConnectionError):
                back = False
            c.close()
            time.sleep(0.2)
        check(back, "the address gets its slots back once its connections have ended")
        time.sleep(0.3)

    # ---- 9. a request that starts and never finishes
    s5, typ, p = try_hello(a, token, fp, isa)
    if typ == ACK:
        s5.sendall(b"S")
        dt = read_until_closed(s5, a.timeout_ms / 1000.0 + 6)
        check(dt is not None and dt < a.timeout_ms / 1000.0 + 3, "a session that sends one byte of a request and stops is dropped after the timeout",
              "%s (timeout %d ms)" % ("still open" if dt is None else "%.1f s" % dt, a.timeout_ms))
    else:
        check(False, "a session for the trickle check", p.decode(errors="replace"))
    s5.close()

    # ---- 10. a session that says nothing at all
    if a.idle_ms > 0:
        time.sleep(0.3)
        s6, typ, p = try_hello(a, token, fp, isa)
        if typ == ACK:
            dt = read_until_closed(s6, a.idle_ms / 1000.0 + 6)
            check(dt is not None and dt < a.idle_ms / 1000.0 + 3, "a session that sends nothing is dropped after the idle limit",
                  "%s (idle limit %d ms)" % ("still open" if dt is None else "%.1f s" % dt, a.idle_ms))
        else:
            check(False, "a session for the idle check", p.decode(errors="replace"))
        s6.close()
    else:
        print("SKIP  idle limit (give --idle-ms to run it)")
    print("%s: %d failure(s)" % ("OK" if fails == 0 else "FAILED", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
