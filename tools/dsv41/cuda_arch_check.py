#!/usr/bin/env python3
"""Does this nvcc build these CUDA architectures?  (BBXL, docs/DEEPSEEK41_SPLIT.md "Blackwell (sm_120) builds")

Used by tools/dsv41/build_gpu_server_linux.sh and build_gpu_server_windows.ps1 before they touch CMake; the same table as
cmake/strata_cuda_arch_check.cmake.  No nvcc is needed to test it: pass --nvcc-version.

  cuda_arch_check.py --arch "86-real;120" --nvcc /usr/local/cuda-13.0/bin/nvcc
  cuda_arch_check.py --arch 120-real --nvcc-version 12.4        -> exit 2, says what to install
Exit 0 ok (warnings on stderr), 2 refused, 3 could not read the nvcc version.
"""
import argparse
import re
import subprocess
import sys

# (first architecture, nvcc that has it)
NEEDS = [(121, (12, 9)), (100, (12, 8))]
# 13.2.0 / 13.2.1 miscompiled the engine for sm_120 (spark/build_engine.sh, docs/TROUBLESHOOTING.md; 13.2.2 is fixed)
BAD_SM120 = [(13, 2)]


def parse_version(text):
    m = re.search(r"release (\d+)\.(\d+)", text) or re.search(r"V(\d+)\.(\d+)", text) or re.search(r"^(\d+)\.(\d+)", text.strip())
    if not m:
        raise ValueError("no CUDA version in %r" % text[:80])
    return int(m.group(1)), int(m.group(2))


def bases(arch):
    out = []
    for a in re.split(r"[;, ]+", arch.strip()):
        b = re.sub(r"-.*$", "", a)
        if b.isdigit():
            out.append((a, int(b)))
    return out


def check(arch, version):
    """-> (errors, warnings)"""
    errors, warnings = [], []
    v = "%d.%d" % version
    for a, b in bases(arch):
        for first, need in NEEDS:
            if b >= first:
                if version < need:
                    errors.append("architecture %s (sm_%d) needs nvcc %d.%d or newer, this toolkit is %s: install CUDA %d.%d+ "
                                  "(13.0 is the release toolkit) or build for the older architectures only (ARCH=86-real)"
                                  % (a, b, need[0], need[1], v, need[0], need[1]))
                break
        if b >= 120 and version in BAD_SM120:
            warnings.append("CUDA %s: 13.2.0 / 13.2.1 miscompiled the engine for sm_120 (13.2.2 is fixed); 13.0 is the release toolkit" % v)
        elif b >= 120 and version < (13, 0):
            warnings.append("sm_%d with CUDA %s: engines built with CUDA 12.8 crashed on long prompts (issues #220, #224); use 13.0 or newer" % (b, v))
    return errors, warnings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, help='CMAKE_CUDA_ARCHITECTURES, e.g. "86-real;120"')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--nvcc", help="path of nvcc (runs `nvcc --version`)")
    g.add_argument("--nvcc-version", help="version text, e.g. 12.8 or 'release 13.0, V13.0.88'")
    a = p.parse_args(argv)
    try:
        text = a.nvcc_version
        if a.nvcc:
            text = subprocess.run([a.nvcc, "--version"], capture_output=True, text=True, check=True).stdout
        version = parse_version(text)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        print("cuda_arch_check: cannot read the nvcc version: %s" % e, file=sys.stderr)
        return 3
    errors, warnings = check(a.arch, version)
    for w in warnings:
        print("cuda_arch_check: warning: " + w, file=sys.stderr)
    for e in errors:
        print("cuda_arch_check: REFUSED: " + e, file=sys.stderr)
    if errors:
        return 2
    print("cuda_arch_check: arch %s ok for nvcc %d.%d" % (a.arch, version[0], version[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
