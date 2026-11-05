# Remote experts: one MoE model across several PCs

Status: working prototype on the `remote-experts` branch of DBN-AIO/strata (forked from Niko1221/Strata v0.2.9).
Goal (BigBrain-XL, DBN-AIO/AIO PR #1434): expert parallelism over the LAN. The weights stay where they are; only the
MoE layer's input (n_tok x n_embd fp32, 10 KB per token for Qwen3.8-Flash-Next) and the expert output rows cross the LAN.

## The cut point

Decode and verify windows run the MoE of every layer through one function:

| What | Where |
| --- | --- |
| router -> top-10 ids per token (GPU, in the captured graph) | `Verifier::run` -> `native_router_top10_multi` (`src/core/verify.cpp`, around line 2199) |
| the pool callback the window calls once per layer (host thread, GPU spin-waits on it) | `PoolMultiFn` (`include/strata/core/verify.hpp`), `drive_pool_multi_impl` (`src/program/generate.cpp`) |
| **expert dispatch: classify each (token, expert) entry as GPU cache hit / PCIe / CPU, compute the CPU ones with the CPU kernels, return the un-weighted rows** | **`expert_pool_dispatch_multi` (`src/core/expert_source.cpp`)** |
| weighted sum over the 10 rows, shared expert, residual | on the GPU after the pool (`moe_combine`), unchanged |

The dispatch already has two helper tiers that "own" entries (`d.peer`, `d.remote[]`: experts on another GPU of the same
PC) and write their result rows into the same `out` buffer. Remote experts over the LAN is a third owner of the same
kind: entries with `kind == -1` (the CPU's) of a handed-out layer get `kind = 3`, are sent to an expert server
(`NetExperts::begin`) before the local CPU work and received after it (`NetExperts::finish`). Nothing after the pool
changes, so the result is the same bits as the local CPU run (verified, below).

The **prompt path** (`src/prefill/prefill.cpp`) goes through the same servers for the chunks it stages after their
routing: "The prompt path through the expert servers" below. Longer chunks (at or above `stream_all_min()`), a layer
split and the fused kernels still read the conductor's local copy, and every path falls back to it when a server fails,
so a conductor keeps its copy of handed-out layers for now.

## Files

| File | Role |
| --- | --- |
| `include/strata/core/net_experts.hpp`, `src/core/net_experts.cpp` | wire protocol, socket shim (POSIX or Winsock behind one `#ifdef`), `NetExperts` (client), `serve_experts` (server loop) |
| `src/core/expert_source.cpp` (`expert_pool_dispatch_multi`) | the hook: `d.net` begin/finish, `d.only_mask` for the server |
| `include/strata/core/expert_source.hpp` | `ExpertDispatch::net`, `ExpertDispatch::only_mask`, `FileExpertSource::resident_bytes` |
| `src/program/generate.cpp` | flags `--expert-server --listen --serve-layers --remote-experts`, server warm-up |
| `tools/slice_gguf.py` | sparse slice of a GGUF shard (header + dense + the served layers' experts) for a host with little disk |

## Use

Server (any host with the engine; AVX2 is enough, the CPU kernels are Strata's own, a GPU is needed only because the
engine initialises CUDA for its dense weights, about 4.3 GB VRAM with `--expert-cache 300`):

```sh
strata --pack PACK --native SHARD1.gguf --max-context 1024 --prefill auto --spec 2 \
       --expert-profile data/expert-profile.bin --expert-cache 300 --mmap-experts --pcie-frac 0 \
       --gpu N --expert-server --listen 192.168.1.130:7840 --serve-layers 38-47
```
(`STRATA_EXPERT_TOKEN` set in the environment of both commands: see "Security".)

Conductor (the normal command plus one flag):

```sh
strata ... --remote-experts 192.168.1.130:7840=38-47[,host2:7840=24-37]
```

- Default port 7840 (several servers on one host: 7840, 7841, ...).
- `--listen` takes `host:port`, `[v6addr]:port` or a bare port, and **binds exactly that address**. The default is
  `127.0.0.1:7840`: reachable from this PC only (the localhost test below). For a conductor on another PC write the
  interface address (`192.168.1.130:7840`) or, deliberately, `0.0.0.0:7840` (every interface). Ports must be 1..65535 and
  `--serve-layers a-b` / `--remote-experts` ranges are digits only: a typo is an error, never a silent 0.
- Layer ranges are inclusive and must not overlap; the handshake checks the protocol version (both directions), n_embd,
  n_expert, n_layers, a **model fingerprint**, the **CPU kernel path** and that the server's range covers the requested one.
  The fingerprint is a SHA-256 over the expert layout (tensor types, per-layer formats, offsets, sizes) and the sizes of
  the model files; a different quantization or file with the same geometry is refused with both fingerprints in the
  message. The kernel path (avx512 / avx512bw / avx2 / baseline, plus an `STRATA_FORCE_ISA` cap) must match too: the AVX2
  and AVX-512 kernels are parity-tested, not proven bit-identical, so a mixed pair is refused unless the server runs with
  `STRATA_EXPERT_ALLOW_ISA_MISMATCH=1` (then the rows are equal within the kernels' tolerance, not bit for bit). The
  `STRATA_*` environment knobs the CPU dispatch reads must also be the same on both hosts.
- `--expert-server` and `--remote-experts` are exclusive. A server always runs the plain CPU pool: a peer GPU, helper GPUs
  and a net route are forced off in server mode (and `serve_experts` refuses to start if any is set).
- Timeouts: every socket read, write and connect on both ends gives up after 10 s (`--expert-timeout-ms N` or
  `STRATA_EXPERT_TIMEOUT_MS`, 100..600000). The server's idle session waits for the next request without a read timeout
  (a conductor may pause between prompts; TCP keepalive notices a dead host), but a connection that has not completed the
  handshake within the timeout is dropped.
- The server holds its experts in RAM: it touches every page of its layers at start (`strata net: served layers ...
  resident ...` line gives the resident GiB). `mlock` is attempted and needs `ulimit -l` headroom; without it the pages
  stay in the page cache and the OS may evict them under memory pressure.
- The server does not load experts of other layers: a request outside its range is refused.
- Use `--pcie-frac 0` on conductor and server for runs that must be reproducible (the PCIe share is otherwise measured
  from CPU timing and changes which experts the GPU computes).

## Security (trusted LAN only)

The expert server runs another host's requests on your CPU and your page cache, and the protocol is plain TCP. Treat it
like an NFS export: **trusted LAN only, firewall port 7840 to the conductor's IP** (`ufw allow from <conductor> to any
port 7840 proto tcp`), and never expose it to the internet.

- **Bind address.** Loopback unless you write another address after `--listen`. A non-loopback bind prints a warning with
  the address, and a second one when no token is set.
- **Shared secret.** Set `STRATA_EXPERT_TOKEN` (any string, 16+ random characters; never put it on a command line) in the
  environment of the server **and** the conductor. The server sends a random 32-byte nonce; the conductor answers in HELLO
  with `HMAC-SHA256(token, nonce || hello fields)`; the server compares in constant time and refuses with a generic
  "authentication failed". The token never crosses the wire and a recorded HELLO cannot be replayed (new nonce per
  connection). A server with a token refuses a conductor without one, and the other way round the conductor warns that the
  server does not ask. What the token does **not** do: encrypt the hidden states, or authenticate the frames after the
  handshake. Anyone who can sniff or hijack the TCP stream can still read or alter it; use a trusted segment or a tunnel
  (`ssh -L`, WireGuard) for anything more.
- **Requests are validated before anything runs.** Every expert id in `[0, n_expert)`, `k` equal to the model's top-k,
  `1 <= n_tok <= the window`, `n_tok*k` within the dispatch tables, arithmetic mode 0/1, mask bytes 0/1, the layer inside
  the served range, the payload size exact. A bad request is answered with ERR and **the connection keeps serving**; the
  dispatch state is reset before each request, so one bad request can no longer disable the server (it did before:
  `d.failed` stayed set until a restart). Frame sizes are capped (4 KiB before authentication, 1 MiB per request).
- **One conductor at a time**, because the dispatch is single-threaded. Handshakes run on their own threads (at most 8 at
  once), so an idle or half-open connection cannot block the real conductor; a second authenticated conductor gets ERR
  "busy".
- **SIGPIPE** is ignored in both modes and sends use `MSG_NOSIGNAL` (`SO_NOSIGPIPE` on macOS): a peer that dies mid-frame
  gives an error, not a dead process.

`tools/net_experts_probe.py` checks all of this against a running server (wrong/missing token, oversized pre-auth frame,
each kind of bad request followed by a good one, busy, idle connection).

## Faults on the conductor

A server that is down, hung, or answers wrongly no longer stops generation. On any network error (send/receive failure or
timeout, a bad or mismatched response, an ERR frame) the conductor drops that route, **computes the rows itself** with its
own CPU kernels (it still holds every expert locally, which prefill needs anyway), logs the outage **once**
("expert server ... lost: ...; computing those experts locally, next try in 1000 ms"), and reconnects on a later call with
exponential backoff (1 s, 2 s, ... 30 s, with a 2 s connect timeout so a dead host costs at most 2 s per retry). The
reconnect repeats the whole handshake (a server restarted with another model is refused). A request that is abandoned
between send and receive (an error elsewhere in the dispatch) also drops the route, so the stream can never be left one frame
behind. The exit summary counts outages, reconnects and locally computed rows. Output is the same bits either way, because the
local rows use the same CPU kernels; only the speed changes.

## Wire protocol "SXP1"

See the header comment of `include/strata/core/net_experts.hpp`: length-prefixed frames, little endian (asserted at compile
time, with the header and message sizes and IEEE-754 floats), one persistent TCP connection, `TCP_NODELAY`; version 2:
CHALLENGE (server nonce), HELLO (version, geometry, kernel path, model fingerprint, HMAC), HELLO_ACK (version, served
layers, top-k, kernel path), REQ (layer, n_tok, k, arithmetic, ids, mask, x), RESP (rows of the masked entries in order), ERR,
BYE. Both sides check the version; a version-1 peer is refused with a clear message. Hidden states are fp32 so the result is bit identical to the local CPU run; fp16/bf16
on the wire would halve the traffic (about 20 KB out, 90 KB in per call at 4-token windows) and is a later option.

## Windows (stream PC: Ryzen 9 5950X, RTX 5070 Ti)

Built and run 2026-11-03 on the stream PC as `strata-expert-server.exe`, the CPU-only server (no CUDA toolkit is installed there
and none is needed: it links no CUDA library). The socket shim uses Winsock (`ws2_32` is linked by `cmake/cpu_expert_server.cmake`;
`WSAStartup` once, `WSAPoll` with `POLLRDNORM` / `POLLWRNORM`, `NOMINMAX` and `_WIN32_WINNT=0x0A00` set before `winsock2.h`); the
warm-up `mlock` is skipped (the pages are touched, the log says `mlocked 0.00 GiB`). The CUDA engine's own `--expert-server` on the
5070 Ti (sm_120) is not built: it needs the CUDA 13.x toolkit (Jack: NVIDIA CUDA Toolkit 13.0-13.2 installer, "Visual Studio
integration" off), then the engine build of docs/SOURCE_INSTALL.md "From source: Windows".

Toolchain used: Visual Studio Build Tools 2026 (MSVC 14.51, x64), the CMake and Ninja that ship with / beside it. All on C:.

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
cmake --build build-cpu --target strata-expert-server -j 6
```

Configure 55 s (it fetches the pinned llama.cpp), build about 2.5 minutes. On the stream PC run both through the capped CPU lane
(`AIOrchestrator/tools/cpu_lane/win_job.py`, 70 % cap; it refuses in streamer mode).

Model files: the server needs only a sparse slice of each GGUF shard (tools/slice_gguf.py `--experts-only`) with the real shard sizes
and `native_experts.txt` + `index.txt` of the pack. A sparse file does not survive `scp` / `tar` to NTFS (it would be filled with
zeros: 66 GB). Copy only the data extents and set the NTFS sparse flag first (`fsutil sparse setflag`); do not grow the file with
Python's `truncate()` (it zero-fills on Windows): seek to size-1 and write one byte. Layers 38-43 are 3.96 GiB of data.

Run (never from an agent or ssh shell: it would die with that app's job object): `tools/stream_pc/start_expert_server.ps1` registers a
one-shot Task Scheduler task whose hidden `pythonw` launcher starts the server detached, then removes the task (the same pattern as
TabbyAPI's `switch_local.py`). `STRATA_EXPERT_TOKEN` comes from a file, not a command line. Firewall: 7840-7843 inbound from the
conductor only (rule `AIO BigBrain-XL expert server` on the stream PC). Stop with the PID in `<log>.pid`. See tools/stream_pc/README.md.

### Measured (2026-11-03, Windows server + Linux conductor)

Server: stream PC (5950X, AVX2, busy with the stream PC's usual services, TabbyAPI on the 5070 Ti), layers 38-43, 15 pool workers,
over the house LAN (2.5 GbE). Conductor: gaming PC, Linux (nobara), RTX 3090 GPU 0 (5664 MiB), `bin-tip`, `--adapt-every 0`,
`--pcie-frac 0`, `--spec 2`, greedy, 14 prompt tokens, 64 new tokens, shared token, `--remote-experts 192.168.1.27:7840=38-43`.

| Run | decode tok/s | calls | round trip | server compute | output (md5 of the token line) |
| --- | ---: | ---: | ---: | ---: | --- |
| single process, cold | 3.41 | - | - | - | `9646d49f` |
| single process, warm | 30.02 | - | - | - | `9646d49f` |
| Windows server, run 1 | 15.12 | 252 | 7.77 ms | 6.48 ms | `9646d49f` |
| Windows server, run 2 | 13.52 | 252 | 7.84 ms | 6.57 ms | `9646d49f` |
| Windows server, run 3 | 4.93 (conductor box loaded) | 270 | 8.12 ms | 6.76 ms | `9646d49f` |

The 64 tokens are identical in all five runs (same CPU kernels, avx2 on both ends, same bytes). The prompt path went through the
Windows server too (333 experts, 12 windows, 0.9 MB out / 8.2 MB in, 221-236 ms). Per decode call 23 KB out, 187 KB in; wire and
framing are 1.2-1.4 ms of the round trip, the rest is the server's compute: 6.5 ms for 6 layers' rows is about 4x the T340's
1.5 ms, because the stream PC is shared with the stream (TabbyAPI, ComfyUI, OBS-side services) while it computes. Resident: working
set 2.7 GB of the 3.96 GiB mapped (Windows trims mapped pages; they stay in the standby list), private 0.26 GB, 0 MiB of VRAM (the
5070 Ti's 10.7 GB used by TabbyAPI did not move). `probe`: `tools/net_experts_probe.py` passes all checks against the Windows server
(wrong / missing token, oversized frame, 11 kinds of bad request, busy, idle connection). Known gap: the log line
`page-cache resident` reads 0.00 of 0.00 GiB on Windows (`resident_bytes` has no Windows implementation).

## CPU-only host (T340)

`strata-expert-server` is the engine's expert server without the engine: it needs no GPU, links no CUDA library and never
initialises CUDA. It is built by configuring with `STRATA_ENABLE_CUDA=OFF` (the default; also no HIP/SYCL), on x86-64 Linux:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF      # add -DSTRATA_BUILD_TESTS=ON for the CPU tests
cmake --build build-cpu --target strata-expert-server -j8
```

It takes the engine's server command line (the same words, so one script serves both builds); flags that only configure the
GPU engine (`--gpu`, `--expert-cache`, `--expert-profile`, `--spec`, `--pcie-frac`, `--mmap-experts`, ...) are accepted and ignored:

```sh
strata-expert-server --pack PACK --native SHARD1.gguf --expert-server --listen 7840 --serve-layers 38-43 [--pool-workers N]
```

| What | Where |
| --- | --- |
| CMake (option `STRATA_BUILD_EXPERT_SERVER`, default ON when CUDA/HIP/SYCL are off) | `cmake/cpu_expert_server.cmake`, included last by `CMakeLists.txt` |
| the program: geometry from the GGUF, expert layout of the pack, `FileExpertSource` (the GGUF shards mapped in place), CPU `ExpertPool`, `serve_experts` | `src/program/expert_server_main.cpp` |
| the CUDA engine's own `expert_source.cpp` and `net_experts.cpp`, compiled unchanged | `strata_expert_host` library |
| the few CUDA runtime calls those make (page-locked memory, events, streams): every device call answers "no device", which they already handle | `include/strata/cpu_only/cuda_runtime.h` (`STRATA_CPU_ONLY`) |
| the few GPU symbols they name (pinned arena, GPU expert cache, helper/peer GPU, four CUDA kernels): never reached by the server; the kernels abort with a message if they ever are | `src/core/cpu_only_stubs.cpp` |
| tests: `expert_server_cli`, `expert_server_needs_model`, `expert_server_links_no_cuda` (readelf: no `libcuda`/`libcudart`/...) | `cmake/check_no_cuda_link.cmake` |

The rows are the same bits the conductor's own CPU pool would produce: the server runs `expert_pool_dispatch_multi` with no plan,
no residency table and no cache, exactly as the CUDA engine's `--expert-server` path does. A new CUDA call or GPU symbol in
`expert_source.cpp` breaks the CPU-only build on purpose (CI, below); add it to the shim or the stubs after checking that the
server cannot reach it.

The dense weights are the conductor's, not the server's. A server needs only the GGUF header and the experts of its layers, so
`tools/slice_gguf.py SRC DST LO HI --experts-only` (run on every shard) makes a sparse copy of just those: layers 38-43 of the
35 GiB shard 1 are a 3.97 GiB file; shard 2 is its header only (4 KiB). `--pack` needs only `native_experts.txt` (and `index.txt`).

Not covered: the prompt path (as for any server, see "Not covered yet").

### CI

`.github/workflows/build-linux.yml` configures with CUDA off and tests on push to `remote-experts` / `cpu-only-build` and on every
pull request from this repository: configure, build everything, `ctest` (93 tests run of 94 registered: every CPU unit and parity test of the
tree plus the three server tests; five skip themselves on a CPU without AVX-512), and a check that the server links no GPU library. It runs on `runs-on: [self-hosted, linux-fast]`, not
on GitHub-hosted runners, in 2.5 to 7 minutes on the T340 (`timeout-minutes: 30`); nothing is cached on D: (the pinned llama.cpp checkout and
cmake live in `.strata-ci-cache` beside the runner's work folder, on the runner's system disk; the job refuses to start with less than 8 GB free). `isa_guard_check` is excluded: it fails on GCC 15 for two
unlisted `iq_avx512.cpp` functions (`iq512_reference_gu_rows`, `iq512x_rows`), in files this change does not touch; fix the allow-list or the
compiler flags, then drop the exclusion.

**CI runner.** The existing `t340-fast-*` / `gaming-fast-*` runners belong to `DBN-AIO/AIO`; a runner is bound to one repository,
so this repository needs its own. One was registered on the T340 as `t340-strata-1` (labels `self-hosted`, `linux-fast`), started by
hand in `~/strata-runner` (`nohup ./run.sh`), not as a service. To make it permanent: `sudo ./svc.sh install dadbodnerd && sudo ./svc.sh start`
in that folder. This repository is public: a pull request from a fork would run the fork's copy of the workflow on that machine.
The job is skipped for fork pull requests, but the workflow file of such a pull request is the fork's own; set
Settings > Actions > General > "Fork pull request workflows from outside collaborators" to "Require approval for all outside
collaborators" (API: `actions/permissions/fork-pr-contributor-approval`, `approval_policy: all_external_contributors`; today it is
`first_time_contributors`) before accepting outside pull requests.

### Measured (2026-11-03): T340 as the server

Server: T340 (Xeon E-2234, 4 cores / 8 threads, no AVX-512, no GPU, 61 GB RAM, Postgres and the bridge running beside it),
GCC 15.2, `taskset -c 0-7`, layers 38-43 (6 of 48), `--pack` with `native_experts.txt` only. Conductor: gaming PC as above (RTX 3090 GPU 0
with about 8 GB free, `cond.sh` settings, `--pcie-frac 0`, `--remote-experts 192.168.1.133:7840=38-43`), direct TCP (the T340's ufw
allows the house LAN; no SSH tunnel). Both machines were busy with CI jobs and other work (load average 15-22 on the gaming PC),
so the absolute tok/s move with the minute; the reference and the remote runs below alternate.

| Item | Result |
| --- | --- |
| build on the T340 (`-j8`, `nice`) | configure 47 s (clones the pinned llama.cpp), `strata-expert-server` 2 min 40 s; the rest of the tree (all 91 tests) another 32 s; build tree 0.65 GB |
| server residency, layers 38-43 | 3.96 GiB of experts touched and page-cache resident (3.96 of 3.96 GiB), `mlock` 0.01 GiB (`ulimit -l` is 8 MiB); process peak RSS 4.16 GB (46 MB anonymous, the rest the mapped GGUF pages); 4 threads (3 pool workers + the calling thread); 0 VRAM |
| start-up | 0.3 s to the listening line with the pages cached |
| 64 greedy tokens, static GPU cache (`--adapt-every 0`) | identical in 8 of 8 runs (4 single-process, 4 with the T340): same output line, same md5 |
| decode, static cache | single process 18.4 / 21.4 / 22.5 tok/s (3.6 on the cold first run); with the T340 16.4 / 16.8 / 17.2 / 17.3 tok/s |
| per call (one call per layer per window; 384 calls) | round trip 2.24 - 2.39 ms, server compute 1.45 - 1.56 ms, so about 0.8 ms of wire and framing; 20.1 KB out, 141.9 KB in |
| the rig CPU server of the earlier test (layers 38-47) | 2.92 tok/s, 16.75 ms round trip, 14.32 ms compute: the T340 is about 7x faster per call |

The T340 run is slower than the single process (about 17 against 20 tok/s) because the conductor's own CPU is a 16-core Ryzen
and the T340 four cores; the layers it serves are computed one round trip in series after the router. A server that is not slower
than the conductor, or overlap of the next layer's request (Next steps 3), changes that.

**Determinism.** With the default adaptive GPU tier (`--adapt-every` > 0) a 64-token greedy run is not reproducible even
single-process: of 4 reference runs, 1 differed from the other 3, and 2 of 5 runs with the T340 differed (one of them equal to the
differing reference). The tier swaps experts into the GPU cache on a thread, so which expert is computed by the GPU (rounding
differs slightly from the CPU) depends on timing under load. The remote path is not the cause: with `--adapt-every 0` the output
is the same in all 8 runs. Use `--adapt-every 0` for any identical-tokens test.

## Measured (2026-11-03)

Model: Qwen3.8-Flash-Next ISTA-DASLab Q2_0 (48 layers x 512 experts, top-10, 33.97 GB of experts). Conductor: gaming PC
(Ryzen 9 5950X, 62 GB, RTX 3090 with ~6 GB VRAM while TabbyAPI runs: `--expert-cache 1500`, `--mmap-experts`,
`--pcie-frac 0`, `--spec 2`, greedy, 14 prompt tokens, 64 new tokens). Binary: sm_86, g++ 15.

| Run | decode | notes |
| --- | ---: | --- |
| single process (reference) | 13.10 tok/s | all CPU experts local |
| conductor + server on localhost, layers 24-47 remote | 17.41 tok/s | 1495 calls, round trip 1.009 ms, server compute 0.903 ms |
| conductor on the gaming PC, server on the AI rig (i5-9400, RTX 3090 card for the dense part), layers 38-47 remote, over an SSH tunnel | 2.92 tok/s | 626 calls, round trip 16.75 ms, server compute 14.32 ms, 20.1 KB out / 90.6 KB in per call |

Correctness: the 64 generated tokens are identical in all three runs (`cmp` of the output lines). The localhost and
cross-host runs use the same CPU kernels on the same bytes, so equality is by construction; it is checked, not assumed.

What the numbers say: the wire is not the bottleneck (loopback round trip minus compute is 0.1 ms; across the LAN through an
SSH tunnel it was 2.4 ms). The rig server is slow because its CPU is (6 cores, no AVX-512, TabbyAPI running next to it):
14 ms of CPU per layer call against 0.9 ms on the 5950X. A faster/more idle host, or experts on the host's GPU instead of
its CPU (BigBrain-XL's plan: the rig has 72 GB VRAM), changes this.

Server residency: layers 38-47 on the rig = 6.59 GiB of experts resident in the page cache (all 5120 blobs) + the dense part
mapped from the sparse slice (3.4 GiB); VRAM 4.3 GB (dense weights + 300 cache slots). Layers 24-47 on the gaming PC =
15.8 GiB of experts. The rig has ~10 GB of RAM available while TabbyAPI runs, so about 10-14 of the 48 layers fit there.

Direct TCP 7840 to the rig was still blocked by its ufw when this ran (the SSH tunnel stood in); the protocol is the same.

### After the hardening (2026-11-03, `remote-experts-hardening`)

Same conductor/server pair on the gaming PC (conductor RTX 3090 GPU0 with `--expert-cache 1100`, 5.7 GB of VRAM; server
on GPU1 with 300 cache slots; layers 24-47 remote; 64 greedy tokens; shared token; loopback). Decode tok/s, in run order,
old binary (`c3ee3745`) against the hardened one: before 11.13 (cold page cache), 23.23, 23.29, 23.56; after 18.61, 23.95,
21.81, 22.82. Warm runs: 23.36 against 22.86 mean, inside the run-to-run noise (the adaptive tier moves a few tok/s);
round trip 1.03-1.15 ms before, 1.08-1.10 ms after. The handshake and the request check cost nothing per call that shows.
The tokens of the eight runs after the first are identical to the single-process reference; the first cold run of the old
binary differs because the adaptive tier swapped experts at other moments (`--adapt-every 0` makes the run
deterministic, which the fault tests below use).

Fault checks (400 and 1000 greedy tokens, `--adapt-every 0`, tokens compared with the single-process run):
- server `kill -9` mid-run: the conductor logs one "lost" line, computes the rows itself, finishes, same tokens;
- server `SIGSTOP` (hung) with `--expert-timeout-ms 2000`: "lost: timed out waiting for the peer" after 2 s, local rows, then
  "is back" once the server resumed; same 1000 tokens as the local run (75,695 rows computed locally during the outage);
- `tools/net_experts_probe.py`: 23 checks pass (wrong/missing token, oversized pre-auth frame, 11 kinds of bad request each
  followed by a good one, busy, idle connection).

## Expert servers on GPU (--expert-server-gpu)

`--expert-server-gpu` (it implies `--expert-server`) lets an expert server hold its served layers' experts in VRAM and
compute them with the GPU kernels instead of the CPU. It uses the helper GPU tiers that already exist
(`--expert-cache-device1..3`, `RemoteExperts`, docs/SECOND_GPU.md). With the flag:

- the helpers are filled with the served layers only (`netproto::served_pairs`: profile order first, then every other
  served expert), not with pairs from all 48 layers;
- CUDA0's own cache does not keep an expert out of the helpers (a server runs no verify window on CUDA0, so an expert
  it held would only fall back to the CPU); CUDA0 holds the dense weights, `--expert-cache 16` is enough;
- the RAM warm-up touches only the experts no helper holds;
- the server logs how many served experts sit in helper VRAM, and after each session how many requested rows the GPUs
  computed (`9443 rows asked, 9375 computed on helper GPUs (99.3%)`).

Entries no helper holds stay on the server's CPU. **The rows are not bit-identical** to a CPU server's: the GPU kernels
round differently (the same as the conductor's own cache, bench/results/2026-09-27-cache-parity), so the generated text
can differ from a CPU-server run. Without the flag the server stays the CPU-only, bit-identical one described above.

Rig command (GPUs pinned by UUID: CUDA0 = the RTX 3090 for the dense weights, CUDA1..3 = the three RTX 3060):

```sh
strata --pack PACK --native SHARD1.gguf --max-context 1024 --prefill auto --spec 2 \
       --expert-profile data/expert-profile.bin --mmap-experts --pcie-frac 0 \
       --expert-server-gpu --listen 127.0.0.1:7840 --serve-layers 38-47 --expert-cache 16 \
       --gpu GPU-d526893c-...,GPU-1608aa48-...,GPU-4382b370-...,GPU-1bab2dba-... \
       --expert-cache-device1 N1 --expert-cache-device2 N2 --expert-cache-device3 N3
```

Size each `N` from the card's live free memory: while TabbyAPI runs, a card may take only its free memory minus 1 GB,
and a 3090 at most 6 GB in total. With about 1.36 MB per slot plus about 400 MiB of context and scratch,
`N = (free_MiB - 1024 - 400) / 1.36`. Measured on the rig: CUDA0 (3090) +3924 MiB, each 3060 +2135..2206 MiB, all
under their caps; 4485 of the 5120 experts of layers 38-47 (5.77 GiB) fit. The RTX 5070 (sm_120) is not used: the
binary is built for sm_86, and the engine takes at most three helper devices.

### Measured (--expert-server-gpu)

Qwen3.8-Flash-Next ISTA Q2_0, 14 prompt tokens, 64 new tokens, greedy, `--spec 2`, `--pcie-frac 0`, layers 38-47 on the
rig over an SSH tunnel (the rig's ufw blocks 7840; the server listens on 127.0.0.1). Conductor on the gaming PC,
`--expert-cache 1100` (5664 MiB of VRAM; the 1500 slots of the first measurement peaked at 6172 MiB, over the 6 GB cap).
Static cache (`--adapt-every 0`), so a configuration gives the same text every run. Page cache warm.

| Rig server | decode tok/s | calls | round trip | server compute | rows on GPU | tokens per round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU (i5-9400, AVX2) | 11.44 | 420 | 5.814 ms | 2.541 ms | 0 % | 1.52 |
| GPU (3 x RTX 3060) | 11.30 | 600 | 3.029 ms | 0.520 ms | 99.3 % | 1.07 |
| GPU, the first run of the session | 3.11 (the conductor's page cache cold) | 600 | 2.955 ms | 0.508 ms | 99.3 % | 1.07 |

- Per layer call the GPU server computes 4.9x faster (0.51 against 2.54 ms) and the round trip falls 48 %
  (3.0 against 5.8 ms); what remains is the SSH tunnel and the frame copies. Remote time per decode round: CPU 58 ms,
  GPU 30 ms.
- Decode tok/s did not move, for a reason outside the server: the GPU server's rounding changed the text, and the new
  text's drafts were accepted less often (1.07 against 1.52 tokens per round: 60 rounds instead of 42). Per round the GPU
  run is faster. With the adaptive cache (the default; measured on the build before the hardening merge, same B2 logic) the
  same pattern: GPU 17.82 / 11.96 tok/s (2.6 ms round trip), CPU 17.21 / 16.35 (4.7 ms).
- Against the 2.92 tok/s of the first rig measurement: that run had 14.3 ms of rig CPU compute per call (TabbyAPI busy
  beside it); in these runs the rig CPU took 2.5 ms. The GPU path does not depend on how busy the rig's six cores are.

## The prompt path through the expert servers

Prefill used to read every expert of a handed-out layer from the conductor's own copy (page cache or SSD) and stream it
to the GPU. Now, for the layers `--remote-experts` hands out, the prompt path sends those layers' non-resident experts
to the expert servers too:

- Where: the host-grouped MoE of `src/prefill/prefill.cpp` (chunks staged after their routing: below
  `stream_all_min()` tokens, one GPU, not the fused path). It reuses the CPU share's slot: the layer's MoE input goes to
  the host, every routed expert the conductor's GPU cache does not hold becomes an "off-GPU" expert, and their
  unweighted rows land in Dm's tail, where the combine weights them like any other row. Nothing of those experts is
  read or streamed on the conductor (`experts streamed` drops by their count).
- Wire: `NetExperts::prompt` sends the chunk in windows of at most `kWindowMaxT` tokens (the server's limit; the windows
  come from `netproto::prompt_windows`), as ordinary REQ frames with the window-invariant arithmetic (`arith 1`), so a
  row does not depend on which window it travelled in. The server is unchanged.
- Faults: a down or failing server is handled as in decode (dropped, retried after its backoff). The prompt does not
  fail: that layer's experts are computed on the conductor's CPU from its own copy (the CPU share's code, split into
  jobs of at most MAXT rows), and the log says so (`prefill: layer N: ...; its K experts computed on this host`).
- Chunks the path does not cover (long prompts at or above `stream_all_min()`, a layer split, the fused kernels) still
  read their experts locally, as before.
- `STRATA_NET_PREFILL`: `1` (default) the servers; `0` off (stream locally, the old path); `2` the reference arm: the
  same experts on this host's CPU share kernels with the same arithmetic, for a bit-for-bit A/B.
- Logged: `strata net: prompt path: 604 experts of 10 layer-chunks computed by the expert servers (189.0 ms of round
  trips); prompt 14.9 tok/s`, and per server `prompt path: 20 windows, 1341 rows, 189.0 ms (server compute 84.4 ms),
  1.4 MB out / 13.7 MB in`.

### Measured (prompt path)

Same model and conductor as above (`--adapt-every 0`, layers 38-47 on the rig, SSH tunnel). Greedy, 64 new tokens.

Bits: with a CPU expert server, prompt rows from the server (`=1`) and the reference arm (`=2`, this host's CPU, same
experts, same arithmetic) give **the same 64 tokens** in every run: 2 + 2 runs with the 14-token prompt and 1 + 1 with
a 119-token prompt. Against the old path (`=0`, the experts streamed to the conductor's GPU) the text can differ: the
GPU's prompt kernels round differently from the CPU's (the same caveat as the CPU share).

| Prompt | path | prompt tok/s | experts read on the conductor | round trips |
| --- | --- | ---: | ---: | ---: |
| 14 tokens | local, streamed to the GPU (`=0`) | 14.6 / 14.2 | 2501 | - |
| 14 tokens | GPU expert server (`=1`), one window at a time | 14.1 | 1899 | 183 ms, 20 windows |
| 119 tokens | local, streamed to the GPU (`=0`) | 69.8 | 5568 | - |
| 119 tokens | GPU expert server (`=1`), one window at a time | 40.7 | 4442 | 1476 ms, 150 windows |
| 119 tokens | reference arm, this host's CPU (`=2`) | 24.7 | 4442 + 1128 on the CPU | - |

(Measured on the build that also has the request pipelining, with `STRATA_NET_PIPELINE=1`: one window at a time, which
is this change on its own. The bit-equality runs used a CPU server; pipelining does not change any row.)

- The conductor no longer reads the handed-out layers' experts for a prompt (602 of 2501 blobs for 14 tokens, 1126
  of 5568 for 119), which is what lets a conductor hold only its own layers.
- Prompt speed: equal for a short prompt; for 119 tokens the servers are slower than the local GPU stream here
  (41 against 70 tok/s): 111 MB of fp32 rows come back through the SSH tunnel for the 10 layers, one window at a time.
  The levers: windows in flight (request pipelining), fp16 rows on the wire, a direct TCP path (the rig's ufw).

## Request pipelining across hosts

A layer can be served by several hosts at once, each owning a range of its experts:

```sh
# conductor: layers 38-47, experts 0-255 on the rig, 256-511 on a second host
strata ... --remote-experts 192.168.1.130:7840=38-47/0-255,192.168.1.30:7850=38-47/256-511
# each server says which experts it owns
strata ... --expert-server-gpu --serve-layers 38-47 --serve-experts 0-255 ...    # rig
strata ... --expert-server     --serve-layers 38-47 --serve-experts 256-511 ...  # second host
```

- `host:port=lo-hi/e0-e1` (`netproto::parse_route_item`; without `/e0-e1` the host owns every expert, as before).
  Two items may share layers only with disjoint expert ranges (`routes_conflict`): every (layer, expert) has one owner.
- Decode: `NetExperts::begin` sends the layer's request to every host that owns one of its CPU entries, all before the
  conductor's own CPU work; `finish` reads every answer. The hosts compute their shares of the same layer at the same
  time, so a layer waits for the slowest host instead of one host doing all of it (it cannot overlap layers: layer
  l+1's input is layer l's output). Entries no host owns stay on the conductor's CPU. A host that fails gives back only
  its own entries (`owned()`), which the conductor computes itself; the other hosts' answers are still read, so their
  streams stay in step.
- Server: `--serve-experts e0-e1` fills the helper GPUs and the RAM warm-up with those experts only, and a request that
  asks for another expert is refused (`parse_req`, ERR; the conductor then computes those rows itself).
- Prompt path: a layer goes to the servers only when its hosts together own every expert (`layer_covered`); each host
  gets its share of every window, round-robin over the hosts, with up to `STRATA_NET_PIPELINE` windows (default 4)
  sent before the first answer is read (bounded, so neither side's socket buffers fill). `STRATA_NET_PIPELINE=1` is the
  old one-window-at-a-time.
- Logged: `strata net: decode: 410 layer calls, 410 with 2+ hosts in flight at once; mean wait for the answers after the
  local CPU work 2.299 ms`.

### Measured (request pipelining)

Same model and conductor (`--adapt-every 0`). Rig: the GPU expert server over the SSH tunnel. Second host: an expert
server on the gaming PC itself (loopback, CPU experts, dense weights on its second 3090), so it shares the conductor's
CPU; there was no other free host to put it on.

Decode, 14-token prompt, 64 tokens (all three gave the same text):

| Layers 38-47 on | decode tok/s | layer calls | 2+ hosts at once | wait after the local CPU work |
| --- | ---: | ---: | ---: | ---: |
| the rig only (all 512 experts) | 11.86 / 17.45 | 420 / 410 | 0 | 3.345 / 3.232 ms |
| rig 38-42 + gaming PC 43-47 (layer split, possible before) | 4.29 / 3.45 | 420 | 0 | 3.835 / 3.558 ms |
| rig experts 0-255 + gaming PC 256-511 (this change) | 15.26 / 15.05 | 410 / 420 | all | 2.299 / 2.146 ms |

- With the experts split across two hosts the conductor waits 33 % less per layer (2.22 against 3.29 ms): each host
  answers half the rows, at the same time (rig round trip 2.0 ms, 102 KB back instead of 187 KB). The rig's helper GPUs
  then hold all 2560 of its experts (100 % of its rows on GPU).
- Decode tok/s is within the noise of these runs (the conductor's own work dominates; the second host competes with
  it for the gaming PC's cores: in the layer-split runs that host took 3.9-4.0 ms per call and the conductor slowed to
  4.3 / 3.5 tok/s).

Prompt path, windows in flight (`STRATA_NET_PIPELINE`), round-trip time of the 10 handed-out layers:

| Prompt | hosts | 1 window at a time | 4 in flight (default) |
| --- | --- | ---: | ---: |
| 14 tokens | rig | 183 ms | 152 ms |
| 119 tokens | rig | 1476 ms (40.7 tok/s) | 991 ms (49.0 tok/s) |
| 14 tokens | rig + gaming PC (experts split) | - | 105 / 83 ms |
| 119 tokens | rig + gaming PC (experts split) | 888 ms | 917 ms |

- On one host, four windows in flight cut the prompt's round-trip time by a third for 119 tokens (prompt 41 -> 49 tok/s).
- With the second host on the conductor's own CPU, that host is the bottleneck (630 ms of compute for its 150 windows),
  so the window depth no longer matters there, and its CPU work also slowed the conductor's own expert streaming
  (prompt 26 tok/s). A second host that is a separate machine does not have that cost.

## Full-stack run (2026-11-03): gaming PC + rig GPU + T340 CPU

The `remote-experts` tip (hardening #3, CPU-only server #2, GPU servers #4, prompt path #5, pipelining #6) on three hosts.
The conductor runs on the gaming PC: RTX 3090 GPU0 pinned by UUID, `--expert-cache 1100`, 5664 MiB of VRAM.
The rig runs `--expert-server-gpu`: 3090 for the dense weights and 3 x RTX 3060 helpers, by UUID. The T340 runs
`strata-expert-server`, the CPU-only build, under `nice -n 19` on its default pool (3 workers + the host thread,
Xeon E-2234, AVX2); its sparse slice holds layers 38-43. Both servers listen on 127.0.0.1: the conductor reaches the rig
through `ssh -L` and the T340 through `ssh -R` from the T340. Every run used `--adapt-every 0`, `--spec 2`,
`--pcie-frac 0`, greedy and 64 new tokens, with two fixed prompts: 14 tokens ("What is the capital of France?" in the
chat template) and 119 tokens (the same question 16 times in one turn). The page cache was warm for every number below.

**Found and fixed by this run:** after #2 and #3, `strata-expert-server` never set its model identity or top-k. Every
conductor was refused at the handshake ("model fingerprint differs"). The server now computes the identity the way
`generate.cpp` does and sets top-k from the GGUF. `serve_experts` refuses to start without them
(`netproto::serve_cfg_problem`), and `--listen` takes `HOST:PORT`.

### Tokens: identical where they should be

Each output line was hashed with md5, first 8 hex digits:

| Prompt | output | runs |
| --- | --- | --- |
| 14 tokens | `9646d49f` | local only (x2); T340 CPU layers 38-43 with the prompt local (`=0`), through the server (`=1`) and the reference arm (`=2`); rig GPU experts 0-255 + T340 256-511 of layers 38-43 |
| 14 tokens | `52acd8b8` | rig GPU layers 38-47; rig GPU 44-47 + T340 38-43 |
| 119 tokens | `3de9503f` | all seven runs: local (x2), T340 (`=1`, `=2`), rig, rig + T340, split |

- **CPU servers are bit-exact:** the T340 server, the decode rows and the prompt rows all give the local-only text.
  It uses the same CPU kernels (AVX2 on both hosts) on the same bytes.
- **The rig GPU changes the text when layers 44-47 run on its GPUs.** This is GPU rounding, as documented for
  `--expert-server-gpu`. The 14-token prompt sits on a near tie at the second token (374 against 248046). The GPU
  kernels' rows for layers 38-47 or 44-47 tip it to the other branch, and from there the texts differ (1.07 instead of
  1.5 tokens per draft round).
- The same GPU rows for layers 38-43 only (the split run) did not tip it, and no configuration tipped the 119-token
  prompt. This is luck at a near tie, not a guarantee: only the CPU servers are exact.

### tok/s

| Configuration | decode, 14 tok | decode, 119 tok | prompt, 14 tok | prompt, 119 tok | remote round trip (decode) |
| --- | ---: | ---: | ---: | ---: | --- |
| local only | 31.90 | 50.64 | 16.4 | 74.7 | - |
| + T340 CPU (38-43) | 25.57 (prompt local) / 23.56 (14.25 in a third run, T340 at 7.6 ms per call) | 21.30 | 17.0 (local) / 14.8 | 42.2 | 3.3-3.7 ms (14 tok), 13.5 ms (119) |
| + rig GPU (38-47) | 7.12 (other text, see above) | 30.79 | 15.6 | 52.2 | 3.2 ms (0.58 compute) |
| + rig GPU (44-47) + T340 CPU (38-43) | 8.70 (other text) | 9.76 | 11.6 | 19.8 | rig 2.9 ms, T340 12.8 / 35.4 ms |
| layers 38-43 split by expert range: rig GPU 0-255 + T340 CPU 256-511 | 15.62 | 17.31 | 11.9 | 31.1 | rig 2.2 ms, T340 9.1 / 17.5 ms, both in flight every call |

Readings:
- **On this hardware every remote layer costs speed.** The gaming PC holds the whole model in its page cache (62 GB of
  RAM), so local is fastest. What the servers buy is capacity: a model that does not fit one host.
- **The T340 is the slow host.** It took 2-10 ms of compute per call on the 14-token runs and 10-30 ms on the 119-token
  runs (four cores, `nice -n 19`, the live fleet beside it). It decides every configuration it is in: the prompt path's
  round trips were 4.3 s of the 6.0 s prompt with both servers.
- **The expert-range split halves the T340's share and keeps the rig on its GPUs.** It beats putting whole layers on
  the T340 (15.6 against 8.7 decode tok/s, 31.1 against 19.8 prompt tok/s) and gives the local-only text on the
  14-token prompt.
- **The rig GPU is fast per call** (0.5-1.0 ms of compute, about 3 ms round trip through the tunnel). Its decode tok/s
  on the 14-token prompt is low only because the text changed and drafts stopped being accepted. On the 119-token
  prompt, where the text stayed the same, it reached 30.8 tok/s against 21.3 for the T340.

Peak VRAM of our processes: the conductor 5674 MiB (6 GiB cap). The rig 3090 +3924 MiB. Each 3060 +2123 to +2260 MiB,
under its free memory minus 1 GB (free was 3399-3541 MiB).

## DeepSeek-V4.1-Flash (src/dsv41, `strata-dsv41`)

The DeepSeek engine is a separate program (`strata-dsv41`, docs/DEEPSEEK41.md) with its own MoE dispatch, and it uses the same
network split as the Qwen engine: same protocol, same `net_experts.cpp`, same flags. Placement, measurements and the test record are in
docs/DEEPSEEK41_SPLIT.md. Status: CPU expert servers on Linux and Windows, the conductor hook, and a synthetic-model test are done and
tested; the conductor hook is compiled into `strata-dsv41` (CUDA 13.2, sm_86) but was not run end to end (no GPU was free); a GPU expert
server and the prompt path through the servers are not done.

### The cut point

Smallest boundary, and the same one as above: **the layer's FFN input rows and the (token, expert) pairs the GPU does not hold go
out; one un-weighted expert output row per pair (`dim` fp32) comes back.** Everything after it stays on the conductor.

| What | Where (`src/dsv41/ds4_model.cu`) |
| --- | --- |
| attention, router (`gate_w`, top-k, score bias), shared expert, hyper-connections, residual | GPU, unchanged |
| GPU-resident experts (VRAM slots) | GPU, unchanged |
| **the CPU lane of one layer: classify nothing, hand over `ents` (expert, up to 16 tokens, slot0) and the input rows `x`, get `c_yd[(slot0 + i) * dim]`** | **`Impl::cpu_experts(l, x, ents, ldx, ord)`**: since this change a wrapper; the old body is `cpu_experts_local` |
| weighted sum `h_part += w * y` over the slots, `combine`, `hc_post` | after the lane, unchanged |
| callers that go through the wrapper | `layer()` (decode and verify windows, up to 16 tokens) and `serve_lane()` (the chunked prompt path's small windows) |
| callers that stay local on purpose | `tune_lane`, `lane_cpu_ms`, `lane_check`, `calibrate_paths` (they time or check the local CPU lane) |

`cpu_experts` sends the entries of a served layer (`Remote::run`: tokens compacted to `n_tok x dim`, ids `n_tok x k` with `k` = the
model's top-k, mask = the pairs) through `NetExperts::begin / finish`, copies the returned rows to `c_yd`, and runs whatever did not
come back (route down, refused, an expert no route owns, more picks than `k`) on the local lane: **any network error is a local
computation of the same rows**, and the route is retried with backoff as for Qwen. Nothing is sent for the drafter's `mtp.*` layers.
The server computes with the engine's own CPU row kernels (`ds4_gguf_cpu`, `ds4_iq_cpu`, `ds4_tq_cpu`, `ds4_gg_cpu`: int8 MXFP4 rows,
IQ1_M / IQ2_XXS / IQ2_XS / IQ3_XXS / ... through Strata's AVX2 / AVX-512 rows or ggml-cpu), in the same order of operations as
`cpu_experts_iq` / `cpu_experts_mx` (activations quantized once per token, gate/up, swiglu with the model's clamp, quantize h, down), so
the rows are the local lane's bits (below).

### What changed in the shared code

- `NetIdentity` has `arch` (`kArchQwen38 = 0`, `kArchDsv41 = 1`) and `n_embd`. The HELLO's former zero pad carries `arch`; **an older
  qwen38 conductor still sends 0 and a qwen38 server still expects 0, so the wire is compatible**, and a server refuses a conductor of
  the other family at the handshake ("engine family differs: server dsv41, yours qwen38"), before the geometry check.
- The hidden size on the wire is `NetIdentity::n_embd` (5120 here, `cpu::H` for Qwen); the window limits are `ExpertServeCfg::max_tok /
  max_entries` (16 and 128 for dsv41: `kWin`, `kWin x kPicks`).
- `serve_experts(const ExpertServeCfg&, const ExpertComputeFn&)` is the engine-independent server loop; `serve_experts(ExpertDispatch&, cfg)`
  is a wrapper over it (behaviour unchanged). `src/dsv41/ds4_net_proto.cpp` compiles `net_experts.cpp` with `STRATA_NET_NO_QWEN`: the
  protocol, client and loop without the Qwen expert pool, layout and kernels.
- The dsv41 fingerprint (`ExpertMap::identity()`): SHA-256 over the geometry (dim, ffn, layers, experts, top-k, swiglu limit), every layer's
  expert types, file indices, offsets, strides and sizes, and the shard names and sizes. `isa` is 0: the dsv41 CPU kernels give the same bits
  on every ISA tier (checked: AVX2 Zen 3 and Coffee Lake, MSVC and GCC, same row hash).

### Use

```sh
# expert server, any host (no CUDA is touched; the same flags on either binary)
strata-dsv41-expert-server --pack PACK --listen 0.0.0.0:7840 --serve-layers 27-29 [--serve-experts 0-191] [--cpu-threads N] [--no-warm]
strata-dsv41 --pack PACK --expert-server --listen 0.0.0.0:7840 --serve-layers 27-29     # the engine binary, returns before any CUDA call

# conductor: the normal command plus one flag (the same spec as Qwen: host:port=lo-hi[/e0-e1], comma separated)
strata-dsv41 --pack PACK ... --remote-experts 192.168.1.130:7840=27-29,192.168.1.133:7840=30-33,192.168.1.27:7840=34-39
```

`STRATA_EXPERT_TOKEN` on both ends, as for Qwen (docs above, "Security"). `PACK` on a server needs only the pack's two text files and the
shards' experts of its layers: `tools/dsv41/slice_stream.py emit PACK 27-29 | slice_stream.py apply SLICE` writes a sparse slice (real
shard sizes, so the fingerprint is the conductor's; on NTFS the sparse flag is set first). `tools/dsv41/plan_split.py` gives each host a
range of layers from the host budgets and prints the `--remote-experts` line, or refuses with the shortfall (stream mode).

### Tests

`tools/dsv41/synthetic_split_test.py BUILD_DIR` (CI: "DeepSeek split test (synthetic)", about 10 s): a DeepSeek-shaped model with random
weights in the real sets' types (`tools/dsv41/make_synthetic_pack.py`: 8 layers, 32 experts, top-6, shared expert and router in the
program, gate = up type / down type mixed as in the 2.00 bpw set: IQ1_M, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS, Q4_K, MXFP4), 64 greedy tokens
from a 6-token prompt window, in one process and through servers on localhost: tokens **and a hash of every expert row** equal for one
server, two servers (layers 0-3 / 4-7), a partial split (layers 2-5, the conductor runs the rest), every layer's experts over two hosts
(0-15 / 16-31), a server that exits after 40 requests (fallback), a server on a sparse slice; refusals for a qwen38 conductor, a wrong
token, another expert count. `ds4_net_test` is the program (`--mode local|remote --remote SPEC`).

Scratch: `--tmp DIR` (else `$RUNNER_TEMP`, else the build dir) holds the packs in a subdirectory removed on exit, SIGTERM included; the test refuses a
tmpfs/ramfs or a directory with under 1 GiB free (`STRATA_TEST_NEED_GIB`). Servers are waited for by polling their port (30 s), not a fixed
sleep; `STRATA_TEST_SERVER_START_DELAY=SECONDS` delays every server start to prove it.

## NVLink

The two RTX 3090 of the gaming PC are NVLink-bridged (`nvidia-smi topo -m`: NV4, four links, 14 GB/s per link shown).
`cudaDeviceCanAccessPeer` is 1 in both directions, `cudaDeviceEnablePeerAccess` succeeds, `cudaDevP2PAttrAccessSupported`
and `NativeAtomicSupported` are 1. The fork's layer split / split engine does not use it: on NVIDIA it exchanges sums
through pinned host memory and calls no peer API (docs/MULTI_GPU.md "No NVLink or peer-to-peer access is needed",
docs/SPLIT_ENGINE.md). Using NVLink for the stage-to-stage hidden state would be a change in the split exchange, not in
the remote-expert path.

## Next steps

1. Prompt path through the expert servers: done for chunks staged after routing ("The prompt path through the expert
   servers"). Open: the stream-all and fused chunk paths, and a conductor without a local copy (today the fallback needs it).
2. Expert servers on GPUs: done for CUDA helper cards ("Expert servers on GPU" above). Open: the stream PC's 5070 Ti
   (sm_120 build), a fourth helper card, and CUDA0's own spare VRAM (it holds only the dense weights in server mode).
3. Overlap: done as far as the data allows ("Request pipelining across hosts"): one layer's experts split across hosts run
   at the same time, and the prompt path keeps several windows in flight. Layer l+1 cannot start before layer l's output.
4. fp16 hidden states on the wire (halves the traffic). The CPU-only server for hosts without a GPU is done: "CPU-only host" above.
