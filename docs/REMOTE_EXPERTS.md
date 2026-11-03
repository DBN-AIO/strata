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
       --gpu N --expert-server --listen 7840 --serve-layers 38-47
```

Conductor (the normal command plus one flag):

```sh
strata ... --remote-experts 192.168.1.130:7840=38-47[,host2:7840=24-37]
```

- Default port 7840 (several servers on one host: 7840, 7841, ...).
- Layer ranges are inclusive and must not overlap; the handshake checks n_embd, n_expert, n_layers and that the server's
  range covers the requested one.
- The server holds its experts in RAM: it touches every page of its layers at start (`strata net: served layers ...
  resident ...` line gives the resident GiB). `mlock` is attempted and needs `ulimit -l` headroom; without it the pages
  stay in the page cache and the OS may evict them under memory pressure.
- The server does not load experts of other layers: a request outside its range is refused.
- Use `--pcie-frac 0` on conductor and server for runs that must be reproducible (the PCIe share is otherwise measured
  from CPU timing and changes which experts the GPU computes).

## Wire protocol "SXP1"

See the header comment of `include/strata/core/net_experts.hpp`: length-prefixed frames, little endian, one persistent
TCP connection, `TCP_NODELAY`; HELLO/HELLO_ACK, REQ (layer, n_tok, k, arithmetic, ids, mask, x), RESP (rows of the
masked entries in order), ERR, BYE. Hidden states are fp32 so the result is bit identical to the local CPU run; fp16/bf16
on the wire would halve the traffic (about 20 KB out, 90 KB in per call at 4-token windows) and is a later option.

## Windows (stream PC: RTX 5070 Ti, sm_120)

The server path uses no Linux-only call: sockets are behind the shim (`ws2_32` is linked by CMake on `WIN32`);
the warm-up `mlock` is `#if !defined(_WIN32)`. Build as for any Windows CUDA build of Strata (docs/SOURCE_INSTALL.md,
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
pull request from this repository: configure, build everything, `ctest` (94 tests: every CPU unit and parity test of the tree plus
the three server tests), and a check that the server links no GPU library. It runs on `runs-on: [self-hosted, linux-fast]`, not
on GitHub-hosted runners, in under 30 minutes (`timeout-minutes: 30`); nothing is cached on D: (the pinned llama.cpp checkout and
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

## NVLink

The two RTX 3090 of the gaming PC are NVLink-bridged (`nvidia-smi topo -m`: NV4, four links, 14 GB/s per link shown).
`cudaDeviceCanAccessPeer` is 1 in both directions, `cudaDeviceEnablePeerAccess` succeeds, `cudaDevP2PAttrAccessSupported`
and `NativeAtomicSupported` are 1. The fork's layer split / split engine does not use it: on NVIDIA it exchanges sums
through pinned host memory and calls no peer API (docs/MULTI_GPU.md "No NVLink or peer-to-peer access is needed",
docs/SPLIT_ENGINE.md). Using NVLink for the stage-to-stage hidden state would be a change in the split exchange, not in
the remote-expert path.

## Next steps

1. Route the prompt path (prefill) through the expert servers, so the conductor needs no local copy of handed-out layers.
2. Expert servers on GPUs (the rig's 72 GB VRAM, the stream PC's 5070 Ti): load the served layers' experts into VRAM and
   compute them with the GPU kernels, so the host CPU stops being the bottleneck.
3. Overlap: send layer l+1's request as soon as its router output exists (today one round trip per layer, in series).
4. fp16 hidden states on the wire (halves the traffic). The CPU-only server for hosts without a GPU is done: "CPU-only host" above.
