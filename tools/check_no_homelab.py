#!/usr/bin/env python3
"""Fail when a change ADDS a private-range address or a home-lab specific name to this PUBLIC repository.

  python3 tools/check_no_homelab.py --base origin/main          # added lines of `git diff -U0 BASE...HEAD`
  python3 tools/check_no_homelab.py --stdin < some.diff          # a unified diff on stdin (the test uses this)

Only ADDED lines are checked, so text that already exists upstream does not trip it.  Documentation addresses must come from the
reserved ranges (192.0.2.x, 198.51.100.x, 203.0.113.x); real machine settings belong in an untracked local file
(tools/win_server/local.json, tools/dsv41/fleet.json: copy the .example files).  A line that truly needs a match carries
`homelab-ok` in the same line.
"""
import argparse
import re
import subprocess
import sys

# private / link-local IPv4 literals (not the documentation ranges)
PRIVATE_IP = re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3})\b")
# a real GPU uuid (the all-zero placeholder is fine)
GPU_UUID = re.compile(r"\bGPU-(?!0{8}-0{4}-0{4}-0{4}-0{12})[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
NAMES = re.compile(
    r"\b(?:T340|t340|dadbodnerd|nobara|AIO-strata|DBN-AIO/AIO|AIOrchestrator|[Ss]tream PC|gaming PC|5950X)\b|"
    r"/mnt/(?:models|bulk|ai|ai_models|gaming_bulk_models)\b|/srv/(?:aio|strata)\b|[A-Za-z]:\\AIO[-\\]|D:/AIStuff|/home/dadbodnerd"
)
CHECKS = [("private address", PRIVATE_IP), ("real GPU uuid", GPU_UUID), ("home-lab name or path", NAMES)]
SKIP_FILES = ("tools/check_no_homelab.py", "tools/test_check_no_homelab.py")


def added_lines(diff):
    path = None
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            path = path[2:] if path.startswith("b/") else path
        elif raw.startswith("+") and not raw.startswith("+++"):
            yield path, raw[1:]


def scan(diff):
    bad = []
    for path, line in added_lines(diff):
        if path in SKIP_FILES or "homelab-ok" in line:
            continue
        for what, rx in CHECKS:
            m = rx.search(line)
            if m:
                bad.append((path, what, m.group(0)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--stdin", action="store_true")
    a = ap.parse_args()
    if a.stdin:
        diff = sys.stdin.read()
    else:
        diff = subprocess.run(["git", "diff", "-U0", "--no-color", a.base + "...HEAD"], capture_output=True, text=True, check=True).stdout
    bad = scan(diff)
    for path, what, hit in bad:
        print("%s: %s added: %s" % (path, what, hit))
    if bad:
        print("%d home-lab specific item(s) in added lines. Use 192.0.2.x / <model-dir> placeholders and keep real values in an untracked "
              "local file (see tools/win_server/local.example.json)." % len(bad))
        return 1
    print("no home-lab specific text in the added lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
