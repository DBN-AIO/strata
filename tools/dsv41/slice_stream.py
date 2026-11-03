#!/usr/bin/env python3
"""Ship only the experts of some layers of a DeepSeek-V4.1 pack to an expert-server host (docs/DEEPSEEK41_SPLIT.md).

An expert server needs the pack's description (ds4.txt, tensors.txt) and the bytes of the experts of ITS layers at their file
offsets: nothing else.  `emit` reads a full pack and writes a stream; `apply` builds from the stream a slice folder on the host:
the same ds4.txt / tensors.txt (so the model fingerprint is the conductor's), the shard files at their REAL sizes but sparse, with
the served layers' expert bytes in place.  Linux and Windows (NTFS: the sparse flag is set before the file grows; a file grown
by truncate() would be zero-filled there).

  # on the host with the model:                               python3 slice_stream.py emit PACK 26-34 [E0-E1] > layers.stream
  # on the expert-server host (any pipe, e.g. through ssh):   python3 slice_stream.py apply OUT_DIR < layers.stream
  ssh nobara 'python3 slice_stream.py emit PACK 26-34' | python slice_stream.py apply C:\\AIO-strata-model\\dsv41-slice

Disk used on the server = the served layers' experts (about 3.3 GB a layer for the 2.00 bpw set, 5.2 GB for a 3 bit set).
`apply` refuses to write into a folder that already has a different slice (nothing is deleted or overwritten silently).
"""
import json
import os
import struct
import subprocess
import sys

MAGIC = b"DS4SLICE1\n"
CHUNK = 64 << 20


def parse_tensors(path):
    """-> (file names {i: name}, expert tensor lines [(layer, off, bytes, file, raw line)])"""
    files, ex = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("file "):
                _, i, name = line.split(None, 2)
                files[int(i)] = name.strip()
            elif line.startswith("t layers.") and ".ffn.experts." in line:
                p = line.split()
                il = int(p[1].split(".")[1])
                ex.append((il, int(p[4]), int(p[5]), int(p[3]), line, int(p[1].split(".")[4])))
    return files, ex


def read_kv(path):
    kv = {}
    for line in open(path, "r", encoding="utf-8"):
        line = line.rstrip("\r\n")
        sp = line.find(" ")
        if sp > 0:
            kv[line[:sp]] = line[sp + 1:]
    return kv


def emit(pack, lo, hi, out, elo=0, ehi=10**9):
    kv = read_kv(os.path.join(pack, "ds4.txt"))
    files, ex = parse_tensors(os.path.join(pack, "tensors.txt"))
    model_dir = kv["model_dir"]
    sizes = {i: os.path.getsize(os.path.join(model_dir, n)) for i, n in files.items()}
    # ranges to ship: the served layers' (and experts') bytes, merged per file
    rng = {}
    for il, off, nb, fi, _, e in ex:
        if lo <= il <= hi and elo <= e <= ehi:
            rng.setdefault(fi, []).append((off, off + nb))
    merged = {}
    for fi, r in rng.items():
        r.sort()
        m = [list(r[0])]
        for a, b in r[1:]:
            if a <= m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        merged[fi] = m
    total = sum(b - a for m in merged.values() for a, b in m)
    # the description: every expert tensor line (the fingerprint covers all layers), no other tensors
    desc = "".join("file %d %s\n" % (i, n) for i, n in sorted(files.items())) + "".join(x[4] if x[4].endswith("\n") else x[4] + "\n" for x in ex)
    ds4 = "".join("%s %s\n" % (k, v) for k, v in kv.items() if k != "model_dir")
    head = json.dumps({"files": {str(i): {"name": n, "size": sizes[i]} for i, n in files.items()}, "layers": [lo, hi], "bytes": total,
                       "tensors_txt": desc, "ds4_txt": ds4}).encode()
    out.write(MAGIC + struct.pack("<Q", len(head)) + head)
    done = 0
    for fi, m in sorted(merged.items()):
        fd = os.open(os.path.join(model_dir, files[fi]), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            for a, b in m:
                off = a
                while off < b:
                    n = min(CHUNK, b - off)
                    buf = os.pread(fd, n, off)
                    if len(buf) != n:
                        sys.exit("short read of %s at %d" % (files[fi], off))
                    out.write(struct.pack("<IQQ", fi, off, n) + buf)
                    off += n
                    done += n
        finally:
            os.close(fd)
        sys.stderr.write("emit: file %d (%s): %.2f GiB so far of %.2f\n" % (fi, files[fi], done / 2**30, total / 2**30))
    out.write(struct.pack("<IQQ", 0xFFFFFFFF, 0, 0))
    out.flush()


def readn(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            sys.exit("stream ended early")
        b += c
    return b


def apply(out_dir, inp):
    if readn(inp, len(MAGIC)) != MAGIC:
        sys.exit("not a DS4SLICE1 stream")
    (hl,) = struct.unpack("<Q", readn(inp, 8))
    head = json.loads(readn(inp, hl).decode())
    os.makedirs(out_dir, exist_ok=True)
    abs_out = os.path.abspath(out_dir).replace("\\", "/")
    meta = os.path.join(out_dir, "slice.json")
    if os.path.exists(meta):
        old = json.load(open(meta))
        if old.get("layers") != head["layers"] or old.get("files") != head["files"]:
            sys.exit("%s already holds another slice (%s): use an empty folder" % (out_dir, old.get("layers")))
    with open(os.path.join(out_dir, "tensors.txt"), "w", newline="\n") as f:
        f.write(head["tensors_txt"])
    with open(os.path.join(out_dir, "ds4.txt"), "w", newline="\n") as f:
        f.write(head["ds4_txt"] + "model_dir %s\n" % abs_out)
    fds = {}
    for i, info in head["files"].items():
        path = os.path.join(out_dir, info["name"])
        if not os.path.exists(path):
            with open(path, "wb"):
                pass
            if os.name == "nt":   # the flag first: growing a normal NTFS file by seek+write fills the gap with zeros
                subprocess.run(["fsutil", "sparse", "setflag", path], check=True, stdout=subprocess.DEVNULL)
            with open(path, "r+b") as f:
                f.seek(info["size"] - 1)
                f.write(b"\0")
        fds[int(i)] = os.open(path, os.O_WRONLY | getattr(os, "O_BINARY", 0))
    got = 0
    try:
        while True:
            fi, off, n = struct.unpack("<IQQ", readn(inp, 20))
            if fi == 0xFFFFFFFF:
                break
            buf = readn(inp, n)
            os.lseek(fds[fi], off, os.SEEK_SET)
            os.write(fds[fi], buf)
            got += n
    finally:
        for fd in fds.values():
            os.close(fd)
    json.dump({"layers": head["layers"], "files": head["files"], "bytes": got}, open(meta, "w"))
    sys.stderr.write("apply: %.2f GiB of layers %d-%d into %s\n" % (got / 2**30, head["layers"][0], head["layers"][1], out_dir))
    if got != head["bytes"]:
        sys.exit("expected %d bytes, got %d" % (head["bytes"], got))


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "emit":
        lo, hi = (int(x) for x in sys.argv[3].split("-"))
        elo, ehi = (int(x) for x in sys.argv[4].split("-")) if len(sys.argv) > 4 else (0, 10**9)
        emit(sys.argv[2], lo, hi, sys.stdout.buffer, elo, ehi)
    elif len(sys.argv) >= 3 and sys.argv[1] == "apply":
        apply(sys.argv[2], sys.stdin.buffer)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
