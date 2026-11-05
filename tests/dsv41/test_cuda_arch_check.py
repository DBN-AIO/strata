"""Failing-first test for the sm_120 toolkit gate (BBXL).  Needs no nvcc and no GPU:
   python3 -m unittest tests/dsv41/test_cuda_arch_check.py      (from the repo root; the cmake half skips without cmake)"""
import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsv41"))
import cuda_arch_check as c  # noqa: E402

CMAKE = shutil.which("cmake") or os.path.expanduser("~/tools/cmake-3.31.6-linux-x86_64/bin/cmake")


def run_cmake(arch, ver):
    return subprocess.run([CMAKE, "-DCHECK_ARCHS=" + arch, "-DCHECK_NVCC_VERSION=" + ver, "-P",
                           os.path.join(ROOT, "cmake", "strata_cuda_arch_check.cmake")], capture_output=True, text=True)


class PyCheck(unittest.TestCase):
    def test_rig_toolkit_124_refuses_sm120(self):
        errors, _ = c.check("86-real;120", (12, 4))
        self.assertEqual(len(errors), 1)
        self.assertIn("12.8", errors[0])

    def test_124_is_fine_for_sm86(self):
        self.assertEqual(c.check("86-real", (12, 4)), ([], []))

    def test_128_builds_120_with_the_crash_warning(self):
        errors, warnings = c.check("120-real", (12, 8))
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

    def test_130_clean(self):
        self.assertEqual(c.check("86-real;89-real;120", (13, 0)), ([], []))

    def test_132_warns(self):
        _, w = c.check("120", (13, 2))
        self.assertTrue(any("miscompiled" in x for x in w))

    def test_121_needs_129(self):
        self.assertEqual(len(c.check("121", (12, 8))[0]), 1)
        self.assertEqual(c.check("121", (12, 9))[0], [])

    def test_native_is_the_toolkits_business(self):
        self.assertEqual(c.check("native", (12, 4)), ([], []))

    def test_parse(self):
        self.assertEqual(c.parse_version("Cuda compilation tools, release 12.4, V12.4.131"), (12, 4))
        self.assertEqual(c.parse_version("13.0"), (13, 0))
        with self.assertRaises(ValueError):
            c.parse_version("no version here")

    def test_cli_exit_codes(self):
        self.assertEqual(c.main(["--arch", "120-real", "--nvcc-version", "12.4"]), 2)
        self.assertEqual(c.main(["--arch", "120-real", "--nvcc-version", "13.0"]), 0)
        self.assertEqual(c.main(["--arch", "120-real", "--nvcc-version", "garbage"]), 3)


@unittest.skipUnless(os.path.exists(CMAKE), "no cmake")
class CmakeCheck(unittest.TestCase):
    def test_refuses_sm120_on_124(self):
        r = run_cmake("86-real;120", "12.4.131")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("needs nvcc 12.8", r.stderr)

    def test_accepts_sm86_on_124(self):
        self.assertEqual(run_cmake("86-real", "12.4.131").returncode, 0)

    def test_accepts_sm120_on_130(self):
        self.assertEqual(run_cmake("86-real;89-real;120", "13.0.88").returncode, 0)


if __name__ == "__main__":
    unittest.main()
