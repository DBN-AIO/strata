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

Not covered yet (needs its own cut): the **prompt path** (`src/prefill/prefill.cpp`) streams expert blobs to the GPU or
computes them on the CPU itself; it reads the conductor's local copy of the experts. Until it is routed through the
same servers the conductor still needs every expert (page cache or SSD) for prompts, even for handed-out layers.

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

## Windows (stream PC: RTX 5070 Ti, sm_120)

The server path uses no Linux-only call: sockets are behind the shim (`ws2_32` is linked by CMake on `WIN32`; `WSAStartup` is
checked and initialised once, thread safe; `WSAPoll`, `ioctlsocket`, `SO_RCVTIMEO` in milliseconds); the warm-up `mlock` is
`#if !defined(_WIN32)`. Build as for any Windows CUDA build of Strata (docs/SOURCE_INSTALL.md,
"From source: Windows": Visual Studio 2022 Build Tools + CUDA 13.0-13.2), then run the same command line with
`strata.exe`. Not built or run on the stream PC from this branch (it is the streaming machine and its D: drive is
failing); keep it buildable.

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

Not covered: Windows (the Winsock branch of the socket shim and the `mlock` guard are untouched, the CMake target is UNIX-only
today; a compile check is a commented stub in `build-linux.yml`), and the prompt path (as for any server, see "Not covered yet").

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

## NVLink

The two RTX 3090 of the gaming PC are NVLink-bridged (`nvidia-smi topo -m`: NV4, four links, 14 GB/s per link shown).
`cudaDeviceCanAccessPeer` is 1 in both directions, `cudaDeviceEnablePeerAccess` succeeds, `cudaDevP2PAttrAccessSupported`
and `NativeAtomicSupported` are 1. The fork's layer split / split engine does not use it: on NVIDIA it exchanges sums
through pinned host memory and calls no peer API (docs/MULTI_GPU.md "No NVLink or peer-to-peer access is needed",
docs/SPLIT_ENGINE.md). Using NVLink for the stage-to-stage hidden state would be a change in the split exchange, not in
the remote-expert path.

## Next steps

1. Route the prompt path (prefill) through the expert servers, so the conductor needs no local copy of handed-out layers.
2. Expert servers on GPUs: done for CUDA helper cards ("Expert servers on GPU" above). Open: the stream PC's 5070 Ti
   (sm_120 build), a fourth helper card, and CUDA0's own spare VRAM (it holds only the dense weights in server mode).
3. Overlap: send layer l+1's request as soon as its router output exists (today one round trip per layer, in series).
4. fp16 hidden states on the wire (halves the traffic). The CPU-only server for hosts without a GPU is done: "CPU-only host" above.
