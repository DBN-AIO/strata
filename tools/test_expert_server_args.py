#!/usr/bin/env python3
"""The expert server refuses a mistyped command line instead of guessing (no GPU, no model: `--check-args` parses and exits).

  python3 tools/test_expert_server_args.py [path/to/strata-expert-server]     (default: build-cpu/strata-expert-server)

Before this check `--serve-layers x` was read with atoi() as layer 0 and `--pool-workers x` as 0 (every core), silently.
Registered with ctest as `expert_server_args` (cmake/cpu_expert_server.cmake), so the CI test step runs it.
"""
import os
import subprocess
import sys
import unittest

BIN = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build-cpu", "strata-expert-server"))


def check_args(*args):
    return subprocess.run([BIN, "--check-args", *args], capture_output=True, text=True, timeout=30)


@unittest.skipUnless(os.path.exists(BIN), "no strata-expert-server at %s (build it, or pass its path)" % BIN)
class ExpertServerArgs(unittest.TestCase):
    def test_serve_layers_garbage_is_refused(self):
        for bad in ("x", "5-3", "-1", "3-", "3-5x", "1-2-3", ""):
            r = check_args("--serve-layers", bad)
            self.assertEqual(r.returncode, 2, "--serve-layers %r: %s" % (bad, r.stdout + r.stderr))
            self.assertIn("bad --serve-layers '%s' (want A or A-B)" % bad, r.stderr)

    def test_serve_layers_good_is_accepted(self):
        for good in ("3-5", "7", "0-0"):
            r = check_args("--serve-layers", good)
            self.assertEqual(r.returncode, 0, "--serve-layers %r: %s" % (good, r.stdout + r.stderr))

    def test_pool_workers_garbage_is_refused(self):
        for bad in ("x", "0", "257", "-1", "4x"):
            r = check_args("--pool-workers", bad)
            self.assertEqual(r.returncode, 2, "--pool-workers %r: %s" % (bad, r.stdout + r.stderr))
            self.assertIn("bad --pool-workers '%s' (want 1..256)" % bad, r.stderr)
        self.assertEqual(check_args("--pool-workers", "8").returncode, 0)
        self.assertEqual(check_args("--pool-workers", "256").returncode, 0)

    def test_bad_listen_is_refused(self):   # it printed the error and carried on: the parser's `return 2` was a bool `true`
        r = check_args("--listen", "host:99999")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--listen", r.stderr)

    def test_unknown_flag_is_refused(self):
        self.assertEqual(check_args("--no-such-flag").returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
