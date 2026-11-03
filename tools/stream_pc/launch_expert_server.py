"""Detached launcher for strata-expert-server on Windows (run by a one-shot Task Scheduler task, see start_expert_server.ps1).

Reads the shared secret from a file (never from the command line), starts the server detached with its output in a log
file, writes the PID next to the log and exits.  Usage (pythonw, hidden):
  pythonw launch_expert_server.py EXE LOG TOKENFILE -- <strata-expert-server arguments...>
"""
import os, subprocess, sys

exe, log, tokenfile = sys.argv[1:4]
args = sys.argv[5:] if len(sys.argv) > 4 and sys.argv[4] == "--" else []
env = dict(os.environ)
if tokenfile and os.path.exists(tokenfile):
    env["STRATA_EXPERT_TOKEN"] = open(tokenfile, encoding="utf-8").read().strip()
out = open(log, "ab")
DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW = 0x8, 0x200, 0x08000000
p = subprocess.Popen([exe] + args, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, close_fds=True)
with open(log + ".pid", "w") as f:
    f.write(str(p.pid))
