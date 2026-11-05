#!/usr/bin/env python3
"""slice_stream.py emit --perm (BBXL S3): a slice laid out in the popularity order.  python3 tools/dsv41/test_slice_perm.py

The conductor renumbers its remote calls with the perm file (original expert id -> rank); an expert server finds expert `rank` at
its usual stride, so the slice must hold the bytes of the ORIGINAL expert at the slot of its rank.
"""
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SL, MK = os.path.join(HERE, "slice_stream.py"), os.path.join(HERE, "make_synthetic_pack.py")
NL, NE = 4, 16


def write_perm(path, perm):
    with open(path, "wb") as f:
        f.write(b"DS4PERM1" + struct.pack("<II", len(perm), len(perm[0])))
        for row in perm:
            f.write(struct.pack("<%dH" % len(row), *row))


def tensors(pack):
    t = {}
    for line in open(os.path.join(pack, "tensors.txt")):
        if line.startswith("t layers."):
            p = line.split()
            q = p[1].split(".")
            t[(int(q[1]), int(q[4]), q[5])] = (int(p[4]), int(p[5]))
    return t


class SlicePerm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.pack = os.path.join(cls.td.name, "pack")
        subprocess.run([sys.executable, MK, cls.pack, "--layers", str(NL), "--experts", str(NE)], check=True, capture_output=True)
        cls.t = tensors(cls.pack)
        cls.model = open(os.path.join(cls.pack, "model.bin"), "rb").read()

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def slice_of(self, args, name):
        out = os.path.join(self.td.name, name)
        e = subprocess.run([sys.executable, SL, "emit", self.pack] + args, capture_output=True)
        self.assertEqual(e.returncode, 0, e.stderr.decode())
        a = subprocess.run([sys.executable, SL, "apply", out], input=e.stdout, capture_output=True)
        self.assertEqual(a.returncode, 0, a.stderr.decode())
        return out

    def test_the_slice_holds_the_original_expert_at_its_rank_slot(self):
        rng = random.Random(5)
        perm = []
        for _ in range(NL):
            r = list(range(NE))
            rng.shuffle(r)
            perm.append(r)                      # perm[l][orig] = rank
        pf = os.path.join(self.td.name, "perm.bin")
        write_perm(pf, perm)
        out = self.slice_of(["1-2", "4-9", "--perm", pf], "s1")
        got = open(os.path.join(out, "model.bin"), "rb").read()
        self.assertEqual(len(got), len(self.model), "real shard size, sparse")
        for il in range(NL):
            for orig in range(NE):
                for w in ("w1", "w2", "w3"):
                    so, n = self.t[(il, orig, w)]
                    do, _ = self.t[(il, perm[il][orig], w)]
                    served = il in (1, 2) and 4 <= perm[il][orig] <= 9
                    if served:
                        self.assertEqual(got[do:do + n], self.model[so:so + n], (il, orig, w))
                    else:
                        self.assertEqual(got[do:do + n], b"\0" * n, "not served: (%d, rank %d, %s) stays a hole" % (il, perm[il][orig], w))

    def test_without_perm_nothing_changes(self):
        out = self.slice_of(["1-1", "0-3"], "s0")
        got = open(os.path.join(out, "model.bin"), "rb").read()
        for e in range(4):
            for w in ("w1", "w2", "w3"):
                o, n = self.t[(1, e, w)]
                self.assertEqual(got[o:o + n], self.model[o:o + n])

    def test_a_bad_perm_is_refused(self):
        rows = [list(range(NE)) for _ in range(NL)]
        rows[2][3] = rows[2][4]                 # not a permutation
        bad = os.path.join(self.td.name, "bad.bin")
        write_perm(bad, rows)
        short = os.path.join(self.td.name, "short.bin")
        write_perm(short, rows[:2])
        junk = os.path.join(self.td.name, "junk.bin")
        open(junk, "wb").write(b"NOTAPERM" + b"\0" * 40)
        for f in (bad, short, junk, os.path.join(self.td.name, "nope.bin")):
            e = subprocess.run([sys.executable, SL, "emit", self.pack, "0-3", "--perm", f], capture_output=True)
            self.assertNotEqual(e.returncode, 0, f)
            self.assertIn(b"perm", e.stderr.lower())


if __name__ == "__main__":
    unittest.main()
