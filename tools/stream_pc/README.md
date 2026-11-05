# tools/stream_pc - the expert server on the Windows stream PC

The stream PC (Ryzen 9 5950X, RTX 5070 Ti, Windows 11) is the operator's streaming machine and runs fleet services beside the
expert server. Everything here is for `strata-expert-server.exe`, the CPU-only build (docs/REMOTE_EXPERTS.md, "Windows").

| File | What |
| --- | --- |
| `start_expert_server.ps1` | registers a one-shot Task Scheduler task, runs it, removes it. The process outlives the task. |
| `launch_expert_server.py` | what the task runs (`pythonw`, no console): reads the token from a file, starts the server detached, writes `<log>.pid` |

Never start the server straight from an agent or ssh shell: a process started there lives in that app's Windows job object
and dies with it (measured for TabbyAPI on 2026-10-06).

## What the AIO launcher (`BigBrain/tools/xl/strata_launch.py`, Windows path) must do

1. Binary: `C:\AIO-strata\build-cpu\strata-expert-server.exe` (`FLAGS["bin_windows"]`; it is not `strata.exe` and has no `serve`
   sub-command). Build: docs/REMOTE_EXPERTS.md, "Windows".
2. Flags are the engine's server words, not the `--expert-port` / `--expert-map` the launcher assumes today:
   `--pack DIR --native SHARD1.gguf --expert-server --listen 0.0.0.0:7840 --serve-layers 38-43` (`--serve-experts e0-e1` for a
   split layer). One port per server: 7840-7843 are open in the firewall for .30 / .130 / .133.
3. Model files: a sparse slice (tools/slice_gguf.py `--experts-only`) of every shard, on C:, with the real shard sizes (the model
   fingerprint hashes them) and NTFS sparse flag set. `--pack` needs only `native_experts.txt` and `index.txt`.
4. Token: `STRATA_EXPERT_TOKEN` in the environment of the server, from a file (`-TokenFile`), never on a command line.
5. Start: `powershell -NoProfile -File tools\stream_pc\start_expert_server.ps1 -Layers 38-43 -Listen 0.0.0.0:7840`; it prints the pid.
6. Stop: `Stop-Process -Id <pid from the .pid file>` after checking the command line contains the slice path; never by image name.
7. Streamer mode: the server uses 16 threads of the 5950X while it computes (a pool of cores-1 workers); stop it on entry to
   streamer mode as the launcher's `watch` already does (pass `-PoolWorkers 4` to cap it instead).
