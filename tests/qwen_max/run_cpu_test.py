#!/usr/bin/env python3
"""ctest driver for native_rows_test: write a small Max-SHAPED synthetic GGUF (tools/make_synthetic_max.py: 4 trunk blocks + the MTP block,
512 experts, top-10, the Max quantisation types, a tiny hidden size) into a temporary folder and run the test on it.

    run_cpu_test.py PATH/TO/native_rows_test

Exit 77 (ctest: skipped) when numpy is missing, which the generator needs.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy is not installed: the synthetic Max-shaped model cannot be written (skipped)")
        return 77
    with tempfile.TemporaryDirectory(prefix="qmax_") as tmp:
        model = os.path.join(tmp, "synthetic_max_4.gguf")
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_synthetic_max.py"), model, "--layers", "4", "--vocab", "4096"], check=True)
        return subprocess.run([sys.argv[1], model]).returncode


if __name__ == "__main__":
    sys.exit(main())
