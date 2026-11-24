# tools/win_server - the expert server on a Windows host

Launchers for `strata-expert-server.exe` (the CPU-only build) and `strata-dsv41-expert-server.exe` on a Windows machine that
also runs other services (docs/REMOTE_EXPERTS.md, "Windows").

| File | What |
| --- | --- |
| `start_expert_server.ps1`, `start_dsv41_server.ps1` | register a one-shot Task Scheduler task, run it, remove it. The process outlives the task. |
| `launch_expert_server.py` | what the task runs (`pythonw`, no console): reads the token from a file, starts the server detached, writes `<log>.pid` |
| `local.example.json` | the machine-specific settings; copy to `local.json` (untracked, in `.gitignore`) and fill in |

Never start the server straight from an agent or ssh shell: a process started there lives in that app's Windows job object and
dies with it.

## Setup

1. Copy `local.example.json` to `local.json` and set: `python_w` (pythonw.exe), `exe_cpu` / `exe_dsv41` (the built servers),
   `pack_qwen`, `native_qwen` (first GGUF shard), `pack_dsv41`, `log_qwen`, `log_dsv41`, `token_file`, and `listen`
   (this machine's LAN address and port, for example `192.0.2.10:7840`; loopback needs no token).
2. Put a random token of 16+ characters in the file named by `token_file`; use the same value in `STRATA_EXPERT_TOKEN` on the
   conductor. The server refuses to listen on a non-loopback address without one.
3. Model files: a sparse slice (tools/slice_gguf.py `--experts-only`) of every shard, with the real shard sizes (the model
   fingerprint hashes them) and the NTFS sparse flag set. `--pack` needs only `native_experts.txt` and `index.txt`.
4. Start: `powershell -NoProfile -File tools\win_server\start_expert_server.ps1 -Layers 38-43`; it prints the pid. One port per
   server; open it in the firewall to the conductor's address only.
5. Stop: `Stop-Process -Id <pid from the .pid file>` after checking the command line contains the slice path; never by image name.
6. If the host also streams or games, stop the server while that runs, or cap it with `-PoolWorkers N`.
