# DeepSeek-V4.1-Flash across several PCs (BigBrain-XL, "Strata strategy")

Goal (Jack, 2026-11-03: "BigBrain 2.0 XL CAN spread them across the network using the Strata strategy"): run DeepSeek-V4.1-Flash with a bigger
quantization than one PC holds (about 3 bit, about 210 GB of routed experts) by spreading the experts' layers over the gaming PC, the AI rig,
the T340 and the stream PC. Branch `dsv41-remote-experts` (off `remote-experts`). The protocol and the cut point are the Qwen engine's;
what is DeepSeek-specific is written down in docs/REMOTE_EXPERTS.md, "DeepSeek-V4.1-Flash". This file is placement, numbers and the record.

## Where things stand

| | State |
| --- | --- |
| dsv41 expert server on Linux (`strata-dsv41-expert-server`, no CUDA; also `strata-dsv41 --expert-server`) | built and run: nobara (gcc 16), T340 (gcc, Xeon E-2234), rig (the T340 binary, i5-9400) |
| dsv41 expert server on Windows (stream PC) | built with MSVC 14.51 (the capped CPU lane, 32 s) and run; rows bit-identical to the Linux conductor |
| CUDA `strata-dsv41` with `--remote-experts` / `--expert-server` | compiles and links (CUDA 13.2, sm_86, gcc 15); the server mode runs with `CUDA_VISIBLE_DEVICES=""`; **the conductor hook has not run end to end: no GPU was free** (GPU 0/1 belong to the other agent's 2.00 bpw run) |
| synthetic DeepSeek-shaped model, local vs split, 64 tokens | passes (CI step, 15 checks) |
| real weights, real split | **not run**: `/mnt/bulk/strata/models/deepseek41/RESULTS.md` did not exist when this was written (the 2.00 bpw set is on disk, 4 shards, sha256 ok). Tools are ready: `slice_stream.py`, `plan_split.py`, the server flags below |
| GPU expert server for dsv41 (`--expert-server-gpu`, BBXL) | built and measured on the rig RTX 3090 (synthetic pack only): rows within 1.1e-3 of the CPU server's, 8-13x faster a call; **not run on the stream PC 5070 Ti** (no CUDA toolkit there yet, TODO-1276) and not run on real weights; see "GPU expert server" |
| parallel remote calls, conductor work overlapped with the servers (BBXL S4) | done, measured on synthetic packs (T340 + rig): "Parallel remote calls (BBXL S4)"; the real engine has not run it |
| prompt path through the servers (BBXL S5) | host side done and tested (`Remote::prompt`, 32-token windows, a conductor pack without the served layers passes the synthetic test); engine: opt-in `STRATA_DS4_NET_PREFILL=1` sends the prompt down the window path (compiled, not run). **Not done:** the chunked flow itself does not call `Remote::prompt` (needs a running engine). "The prompt through the servers (BBXL S5)" |

## Commands

Build (every host; `strata-dsv41-expert-server` and `ds4_net_test` need no CUDA):

```sh
# Linux (T340: ~/tools/cmake-3.31.6-linux-x86_64/bin/cmake, g++ 15 or newer)
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
cmake --build build-cpu --target strata-dsv41-expert-server ds4_net_test -j8
```

```bat
:: Windows (stream PC), through the capped CPU lane; MSVC 14.51 from Visual Studio Build Tools 2026, all on C:
MSYS_NO_PATHCONV=1 C:\Python314\python.exe "D:/AIStuff/AIO/AIO/AI Projects/AIOrchestrator/tools/cpu_lane/win_job.py" --cwd C:\AIO-strata -- cmd /c C:\AIO-strata\build_dsv41_cpu.bat
:: build_dsv41_cpu.bat: call ...\VC\Auxiliary\Build\vcvars64.bat ; cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF ;
::                      cmake --build build-cpu --target strata-dsv41-expert-server ds4_net_test -j 6
```

**The CUDA build on the stream PC (RTX 5070 Ti, sm_120) once the toolkit is installed** (Jack's TODO: NVIDIA CUDA Toolkit 13.x, "Visual Studio
integration" off). CMake finds it through `CUDA_PATH`, which the installer sets (`packaging\build_dsv41_windows.bat` now defaults
`CUDA_ROOT` to it). CUDA 13.0-13.2 accept **Visual Studio 2022 only**; the PC has the 2026 Build Tools (MSVC 14.51), which CUDA accepts from
13.3. So either install the 2022 Build Tools beside them or take CUDA 13.3+:

```bat
set DSV41_ARCH=120-real
set STRATA_GGML_DIR=C:\AIO-strata\build-cpu\_deps\strata_llamacpp-src
:: CUDA 13.0-13.2 with VS 2022:   (default STRATA_VCVARS is the 2022 BuildTools vcvars64.bat)
:: CUDA 13.3+ with VS 2026:       set STRATA_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat
set BUILD_JOBS=6
MSYS_NO_PATHCONV=1 C:\Python314\python.exe "D:/AIStuff/AIO/AIO/AI Projects/AIOrchestrator/tools/cpu_lane/win_job.py" --cwd C:\AIO-strata -- cmd /c packaging\build_dsv41_windows.bat C:\AIO-strata\build-cuda
```

Result `C:\AIO-strata\build-cuda\strata-dsv41.exe` (the build links `ws2_32` for the socket shim; `--expert-server` on it takes the same flags
and never calls CUDA). The equivalent direct CMake line is `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DCMAKE_CUDA_ARCHITECTURES=120-real
-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe -DCUDAToolkit_ROOT=%CUDA_PATH%`. Not run (no toolkit); the CPU-only build and the Linux CUDA
build (`-DCMAKE_CUDA_ARCHITECTURES=86-real`, gcc 15 as host compiler) are the evidence that the sources build.

Start the stream PC's server outside the agent's job object: `powershell -File tools\stream_pc\start_dsv41_server.ps1 -Pack C:\AIO-strata-model\dsv41-pack -Layers 34-39 -Listen 0.0.0.0:7840`
(the one-shot Task Scheduler pattern; token from `C:\AIO-strata-model\expert.token`; stop with the PID in `<log>.pid`). Port 7840 from .30 / .130 / .133 is open on the stream PC.

Slices (a server needs only its layers' bytes): `python3 tools/dsv41/slice_stream.py emit PACK 34-39 | ssh HOST 'python3 slice_stream.py apply SLICE'`;
for a split layer add the expert range (`emit PACK 20-23 0-191`). The slice has the real shard sizes (sparse), so the fingerprint is the conductor's.

Plan: `python3 tools/dsv41/plan_split.py --gguf-dir DIR` (layer sizes from the GGUF headers) or `PACK`, `--uniform-gb 5.2022` for the 3 bit set,
`--drop stream` for stream mode, `--hosts fleet.json` for other budgets; prints `--remote-experts ...`, exit 3 with the shortfall when it does not fit.

## Measured (2026-11-03)

Program: `ds4_net_test --mode local` on a synthetic pack of the real shape (hidden 5120, expert FFN 2304, top-6, 4 layers x 48 experts, types as
the 2.00 bpw set mixes them: IQ1_M, IQ2_XXS / IQ1_M, IQ2_XS / IQ1_M, IQ2_XXS / IQ2_XS), data resident in the page cache, 64 greedy tokens (256
layer calls: one 6-token prompt window per layer, then single tokens). "Compute" is one layer-call: 6 experts (3 matrices of 5120 x 2304) for
one token on the host's CPU, all threads. The weights are random: timing and bit identity, not quality.

| Host | CPU | threads | compute a call | as server: round trip from the gaming PC | wire (round trip - compute) |
| --- | --- | ---: | ---: | ---: | ---: |
| gaming PC (nobara) | Ryzen 9 5950X | 16 | 1.9 ms (quiet), 3.0 ms (other agent running) | - | - |
| stream PC (Windows) | Ryzen 9 5950X, beside the stream's services | 16 | 2.87 ms as a server; 3.74 ms inside the 70 % CPU lane | 3.75 ms | 0.88 ms |
| AI rig | i5-9400 (6 cores, no AVX-512) | 6 | 4.36 ms local, 4.51 ms as a server | 6.44 ms **through an ssh tunnel** (the rig's ufw lets only the T340 reach 7840-7843; I did not change it) | 1.9 ms with the tunnel, about 1.2 ms direct (estimate) |
| T340 | Xeon E-2234 (4 cores, 8 threads), AIO services running | 8 | 7.2 ms local, 6.44 ms as a server | 7.18 ms | 0.75 ms |

Per call 21.6 KB out and 129.4 KB back (hidden 5120 fp32 x 1 token out, x 6 rows back). `ping`: T340 0.12 ms, rig 0.18 ms, stream PC 0.39 ms.
An IQ3_XXS pack of the same shape costs 2.28 ms on the gaming PC against 1.91 ms for the 2.00 mix (+19 %).

**Bit identity.** The hash of every expert row received (64 tokens, 1656 rows of 5120 floats) is `e7f2d1b3164e44c1` for: the gaming PC's own lane
(gcc 16, `-march=native`), the T340 (gcc, Coffee Lake class), the rig (the T340 binary), the Linux conductor with the **Windows (MSVC) server**,
and the Linux conductor with the T340 and the rig (tunnel) as servers; thread counts 1 to 16 give the same hash. The CUDA engine binary's
`--expert-server` returns the same rows as the CPU-only server (checked on the small synthetic pack).

## Placement

Sizes. 2.00 bpw set (ISTA-DASLab IQ2_XXS, read from the GGUF headers): routed experts 132.07 GB, 40 layers of 2.97-3.93 GB (average 3.30 GB; 8.6 MB for
one expert's three matrices), dense weights 6.2 GB, Engram 111 GB on the conductor's SSD. A uniform IQ3_XXS set (3.06 bpw, the "about 3 bit" case; it
is not published, so this is arithmetic): 13.55 MB an expert, 5.20 GB a layer, **208.1 GB** of experts (dense not estimated: it is the conductor's).

What each host can give to experts (GB; the binding number is the smaller of memory and disk):

| Host | RAM free for the page cache | VRAM free | Free disk | Used here |
| --- | ---: | ---: | ---: | --- |
| gaming PC (conductor) | 52 of 64 | about 35 of 48 (dense 6-8, KV and graphs 5) for the hottest experts of any layer | NVMe, plenty | 87 |
| AI rig | 13 (32 GB, 15 free while the fleet runs) | 0 (TabbyAPI holds 36 of 72 GB); 66 if unloaded, but no dsv41 GPU server yet | 24 | min(13, 20) |
| T340 | 30 (64 GB, AIO services use 22) | none | **17** | min(30, 14) |
| stream PC | 48 (64 GB, about 16 used by the stream) | 5 (TabbyAPI holds 11 of 16) | **27** on C: (D: is failing) | min(48, 24) |

Today's plan (CPU servers only), layers in hop order, `plan_split.py` output:

| 2.00 bpw (132.1 GB) | layers | GB | `--remote-experts` |
| --- | --- | ---: | --- |
| gaming PC (conductor) | 0-26 | 86.8 | (local) |
| rig | 27-29 | 10.4 | `192.168.1.130:7840=27-29` |
| T340 | 30-33 | 13.3 | `192.168.1.133:7840=30-33` |
| stream PC | 34-39 | 21.6 | `192.168.1.27:7840=34-39` |

It fits with 0.2 GB to spare on the conductor, 2.6 GB on the rig, 0.7 GB on the T340, 2.4 GB on the stream PC. **Stream mode stops the stream PC's
server**: its six layers (21.6 GB) have no other host (`plan_split.py --drop stream` exits 3, "6 layers (21.6 GB) have no host"). The launcher must refuse the run, or accept
the conductor reading those layers from its SSD (about 3.3 GB x 6 layers cold, a read of up to 119 MB per token); the conductor's route list without the stream PC
runs as is, because a layer no route owns is the conductor's own lane.

| ~3 bit, IQ3_XXS (208.1 GB) | layers | GB | |
| --- | --- | ---: | --- |
| gaming PC | 0-15 | 83.2 | |
| rig | 16-17 | 10.4 | RAM-limited |
| T340 | 18-19 | 10.4 | disk-limited (3 layers need 15.6 GB, 17 are free) |
| stream PC | 20-23 | 20.8 | disk-limited |
| **no host** | 24-39 | **83.2** | 16 layers |

So **the ~3 bit set does not fit the fleet as it is today**: 83 GB short, and the limit is disk and free RAM, not VRAM (136 GB of VRAM sit mostly in
the fleet's TabbyAPI cards, and dsv41 has no GPU expert server yet). What makes it fit (`plan_split.py --uniform-gb 5.2022 --hosts fleet.json`,
rig as a GPU server with 66 GB VRAM and 80 GB of disk, T340 with 40 GB of disk): gaming 0-15 (83.2), rig 16-30 (78.0), T340 31-35 (26.0), stream 36-39 (20.8): all 40 layers,
with 3.8 GB to spare on the conductor, 1 GB on the rig, 4 GB on the T340 and 3.2 GB on the stream PC; without the stream PC it is 20.8 GB short. Needs, in
order: free disk on the rig (+60 GB) and the T340 (+25 GB) or a NAS mount, the dsv41 GPU expert server, and the rig's TabbyAPI unloaded for the run.

Expected speed (experts only, from the measured numbers; add the conductor's dense/attention time per token, which the single-GPU run in RESULTS.md will give):

| Case | experts per token | tok/s ceiling |
| --- | ---: | ---: |
| 2.00 bpw, plan above, layers one host after the other, CPU lane everywhere | 136 ms (27 x 2.5 local + 6 x (2.9 + 0.9) + 4 x (6.4 + 0.75) + 3 x (4.5 + 1.2)) | 7.4 |
| same, every layer's experts split over all four hosts in proportion to their speed (`HOST=lo-hi/e0-e1`, asked at the same time; the layer waits for the slowest) | about 84 ms (2.1 ms a layer, the rig's wire dominates) | about 12 |
| ~3 bit, partial plan (24 placed layers only) | 95 ms for 24 layers | not meaningful until all 40 are placed |

These are CPU-only upper bounds on the remote work. The conductor's VRAM-resident experts (35 GB, the shipped hot-expert profile) are computed on
the GPU and never leave the conductor, so the real remote share per token is smaller; the DSpark drafter's verify windows share expert reads between
tokens. Neither is measured here. The wire is not the limit (0.75-0.9 ms a layer, 1.2 estimated for the rig); the per-layer compute of the slow hosts is.

## Popularity-aware placement (BBXL S3)

Captain 1's order (2026-11-03): log expert hit counts on a real prompt mix, put the hottest experts in VRAM / RAM, the cold ones on the remotes or the SSD, and let
`plan_split.py` take the popularity file. Today's plan is layer-major (host after host, every expert of a layer on one host); this one is expert-major.

**The hit log already exists.** `strata-dsv41` counts every routed pick of every layer (decode windows and chunked prompts) in the expert profile
(`prof`, uint32 [layers + drafter rows][experts], `--expert-profile FILE`), and `--profile-run IDS` runs a prompt set through it and saves the counts. No engine change was
needed for logging. A FILE that does not exist starts at zero, so the file then holds only that run's prompts (the default `<pack>/expert-profile.bin` is the shipped
calibration profile and would be added to: always pass an explicit new path). It is the same file the conductor's own VRAM / RAM tiering reads (`--expert-profile`), so the
planner and the conductor agree on which experts are hot. The shipped `data/dsv41/profile-2.00.bin` is a real (calibration-mix) count set: 43 rows x 384 experts, the top
quarter of the experts take 54 % of the hits.

```sh
# 1. count (real weights: wait for Captain 1's go).  IDS = the token ids of the prompt mix, the format `--profile-run` reads
strata-dsv41 --pack PACK --expert-profile /var/tmp/mix-counts.bin --profile-run mix.ids
# 2. plan (hosts as in fleet.json or the built-in fleet; --allow-ssd lets the coldest experts stay on the conductor's SSD)
python3 tools/dsv41/plan_split.py --gguf-dir GGUF_DIR --popularity /var/tmp/mix-counts.bin --perm-out perm.bin [--groups N] [--allow-ssd]
# 3. one slice per server, laid out by rank: ranks e0-e1 of the layers the plan names
python3 slice_stream.py emit PACK 0-39 e0-e1 --perm perm.bin | ssh HOST 'python3 slice_stream.py apply SLICE'
# 4. server per route item: strata-dsv41-expert-server --pack SLICE --serve-layers 0-39 --serve-experts e0-e1 ...
# 5. conductor: STRATA_DS4_EXPERT_PERM=perm.bin strata-dsv41 ... --expert-profile /var/tmp/mix-counts.bin --remote-experts <the plan's line>
```

**How the plan is made.** Every (layer, expert) has an expected number of picks per token (its layer's counts scaled to top-k = 6). The pairs are ranked by it, then dealt out
in speed order until each tier's memory (and disk) budget is full: conductor VRAM (`vram_gb`), conductor RAM, remote GPU hosts, remote CPU hosts by call time (compute + wire),
and last, with `--allow-ssd` only, the conductor's SSD. Without `--allow-ssd` a set that does not fit the hosts' memory and disk is refused (exit 3), as today. A flat file (every
layer's counts equal) gives today's plan unchanged. A file that is not the engine's counts (wrong size for layers x experts, cut off, all zero, the Qwen engine's ranked `STRP`
profile, a negative count) is refused with exit 2 ("a ranking is not a frequency": a rank cannot say how much hotter expert 1 is than expert 50).

**What the wire needs, and what was added.** An expert server serves ONE contiguous expert range for all its layers, and a route is `HOST=lo-hi/e0-e1`. A hot set is not a
contiguous range of expert ids. So each layer's experts are renumbered hottest first (the rank), the plan hands out RANGES OF RANKS, and the renumbering is a file
(`--perm-out`, `DS4PERM1`: u32 layers, u32 experts, uint16 rank[layer][original expert]):

| Piece | Change |
| --- | --- |
| conductor | `STRATA_DS4_EXPERT_PERM=FILE`: `net::Remote` sends the RANK of every remote expert (`ds4_net.cpp`; unset = ids as they are, nothing else changes; a file that is not a permutation of every layer, is cut off or is for another geometry makes `--remote-experts` fail at start). The conductor's own experts (ranks below the first route, and any rank no route owns) keep their original ids and run locally. |
| slice | `slice_stream.py emit PACK 0-39 e0-e1 --perm FILE`: `e0-e1` are ranks, and the slice holds the bytes of the ORIGINAL expert at the slot of its rank. Same `tensors.txt`, same fingerprint. |
| server | unchanged: it finds rank r at its usual stride. |
| planner | `plan_split.py --popularity FILE [--perm-out FILE] [--groups N] [--allow-ssd]` (also `--experts N`, `--top-k K`; the pack's `ds4.txt` gives both). |

Rows are bit-identical to the unrenumbered split (synthetic test step 4c: tokens and the hash of every expert row equal to the local run; ranks 0-15 on the conductor, 16-31 on a
slice server). `--groups N` gives each of N layer groups its own ranges (the conductor can then follow layer-to-layer differences), at the price of one server process
per (host, group), because a server's range is the same for all its layers; the default 1 needs one server per host.

**Estimate (CPU kernels, measured per-call costs, weighted by hit share).** A layer costs the slower of the conductor's own work and its slowest remote host (the calls are in
flight while the conductor computes its share). Per-call: the table of "Measured" above, the GPU 0.416 ms a call from the 3090 (S2), 2.9 ms an expert read from NVMe at 3 GB/s.
The baseline is today's layer-major plan on the same hit counts (its conductor VRAM also filled hottest-first). Two things are measured, not modelled: the wire is not the limit,
and the hosts' compute. One thing is assumed: a remote GPU server places whole served layers (first fit in layer order, `ds4_gpu_place.hpp`), not the hottest ranks of each layer.

| Popularity (2.00 bpw, 132.1 GB, today's fleet) | conductor's share of the hits | experts a token, popularity plan | today's plan, same hits |
| --- | ---: | ---: | ---: |
| flat | 65 % | 115.2 ms (= today's plan) | 115.2 ms |
| synthetic Zipf s=0.5 (top quarter 48 %) | 80 % | 38.9 ms | 103.4 ms |
| the shipped profile (top quarter 54 %, real counts) | 88 % | 41.7 ms | 99.4 ms |
| synthetic Zipf s=0.8 (top quarter 67 %) | 89 % | 32.2 ms | 95.3 ms |
| synthetic Zipf s=1.1 (top quarter 84 %) | 95 % | 24.8 ms | 88.6 ms |

Read it with care: most of the gain over layer-major is that the conductor keeps 66 % of the experts and the other hosts answer in parallel; the doc's 84 ms for a uniform
expert split (above, not measured) is the fair "no popularity" comparison for the parallel part, and 41.7 ms is about half of it. These are estimates; the real prompt mix decides.

**Toy, measured (T340, CPU only, loopback).** Real-shape synthetic pack (4 layers x 48 experts, 5120 x 2304, the 2.00 mix of types), a router with a hidden Zipf-like preference
(`ds4_net_test --skew 0.3`: the top quarter of the experts take 51-64 % of the hits), counted with `ds4_net_test --count-out` (the engine's profile format), planned with
`plan_split.py --popularity`, a slice laid out by rank, 128 greedy tokens, one warm-up run not counted, then five runs:

| | expert rows over the wire | decode ms a token (5 runs) | tokens / row hash vs the local run |
| --- | ---: | ---: | --- |
| layer-major (conductor layers 0-1, server layers 2-3) | 1716 (50 %) | 26.8-27.7 (median 27.7) | identical |
| popularity (ranks 24-47 of every layer on the server) | 720 (21 %) | 26.1-29.0 (median 26.8) | identical |

The wire work drops by 58 % and the output is bit-identical; the wall time does not move because on ONE 4-core box the "remote" server competes with the conductor for the same
cores, so there is no parallelism to win. The time gain needs separate hosts: the estimate for this toy's two-host fleet is 8.4 ms against 10.2 ms. (The unique-prefix 2k / 16k
prompts of the order need the real engine: not run.) Commands: `~/max2work/s3_toy.sh`, `s3_toy2.sh` on the T340 (not committed: they hold local paths).

**Not done / needs a go.** (1) The real prompt-mix count (needs the real weights loaded). (2) The engine's own conductor-side VRAM / RAM choice is already popularity-driven by
its `--expert-profile`; pass the same counts there. (3) The CUDA `strata-dsv41` binary was not built here (no CUDA on the T340): the conductor change is in `ds4_net.cpp`, which
the CPU build compiles and `ds4_net_test` exercises, and the engine reaches it unchanged through `net::Remote::run`. (4) A GPU expert server that places the hottest RANKS in VRAM
(not whole layers) would make the remote GPU tier worth more than this estimate says.

## Parallel remote calls (BBXL S4)

Captain 1's order (2026-11-03, #22829): split each layer's experts across hosts and call them at the same time (doc ceiling about 12 tok/s against 7.4
asked one after the other); pipeline the next layer's routing.

**What was already there, and what was added.** The Qwen engine's request pipelining (B5, `NetExperts::begin`) is in `core` and dsv41's `net::Remote` already
called it: `begin` writes the layer's request to EVERY host that owns one of its entries before any answer is read, `finish` reads them in route order, so a
layer waits for the slowest host. Rows are written to fixed positions, so the result is bit-identical to a one-host run. What dsv41 did NOT do: `Remote::run`
was ask-and-wait, so the conductor's own experts (the ranks no route owns) only started once every server had answered.

| Piece | Change |
| --- | --- |
| `net::Remote::begin` / `finish` (`ds4_net.cpp`) | `run` split in two. `begin` sends every route's request and says which items are on their way (`pending[q]`); the caller computes the others NOW; `finish` reads the answers. `run` = `begin` + `finish`, unchanged for every other caller. A host that fails in `finish` leaves its items to the caller, as before. |
| `cpu_experts` (`ds4_model.cu`) | begin, the conductor's own experts (`cpu_experts_local`), finish, then the items of a failed host. Same rows, bit for bit: every entry is computed independently of the others. |
| `STRATA_NET_SERIAL=1` (`net_experts.cpp`) | measurement hook, off by default: the answer is read right after each send, so the hosts of a layer are asked one after the other (the "before" arm). |
| test hooks | `STRATA_DS4_SERVER_DELAY_MS` (a server answers N ms late), `ds4_net_test --overlap`, `--local-delay-ms N` (the conductor's lane costs N ms a layer), `--prompt-salt N` (another prompt: a unique prefix for every timing run). |

**Why the next layer's routing cannot be pipelined.** Layer l+1's input is layer l's output: the weighted sum of the picked experts plus the shared expert and the
residual, then the next layer's attention and norm. Its routing, and so its request, needs that row; nothing of it exists before layer l's answers are in. The
only work that does not depend on the answers is inside the layer: the conductor's own experts (now overlapped) and the shared expert and GPU experts (already
concurrent with the CPU lane). A speculative next-layer request (guessing the experts from the profile, as `ahead_ask` does for the SSD) would only prefetch weights,
and a server's experts are resident already. So the overlap is "as far as the data allows", as for Qwen B5. (With several tokens in a window, as the drafter's verify
windows are, all of a layer's tokens already go in one request.)

**Tests (CI: `tools/dsv41/synthetic_split_test.py`, steps 3c-3e; failing first).** Run against the base binaries the new checks fail (`unknown option --overlap`, no server delay
hook: 7 failures); on this branch all pass (0 failures, whole script). The timing checks inject the delay (30 ms a server, 25 ms for the conductor's lane) so they hold on a loaded host.

| Check | Expected | Measured |
| --- | --- | --- |
| two servers (experts 0-15 / 16-31), 30 ms each, 8 layers, tokens and row hash vs the local run | identical; a token about 8 x 30 ms, not 8 x 60 | identical, 243 ms |
| the same with `STRATA_NET_SERIAL=1` (before) | the sum, 480 ms | 473 ms |
| the conductor's own experts 25 ms a layer, one server 30 ms: asked, then computed | 8 x 55 = 440 ms | 432 ms |
| the same with `--overlap` | the longer of the two, 8 x 30 = 240 ms | 240 ms |
| `--overlap` and the server dies after 40 requests | identical tokens and rows through the local fallback | identical, remote and local rows both used |

**Measured on real shapes (T340 conductor, the T340's server on loopback and the rig's over an ssh tunnel; CPU kernels).** Pack: 4 layers x 32 experts, hidden 5120, ffn 2304,
the 2.00 mix of types (`make_synthetic_pack.py --fast`), 64 greedy tokens, one warm-up run not counted, then five runs each with its own prompt (`--prompt-salt` 1-5, so no
two runs share a prefix). Expert rows are the local run's in every run (row hash equal for all five prompts). `ms/token` is the decode time; the T340's server and conductor
share its cores, so only the two-host arms show a gain that is not an artifact of one box.

| Arm | ms/token, 5 runs | median |
| --- | --- | ---: |
| two remote hosts (T340 0-15, rig 16-31), one after the other (`STRATA_NET_SERIAL=1`, before) | 54.2 / 58.1 / 64.6 / 66.0 / 73.0 | 64.6 |
| the same, asked at the same time (S4 / B5) | 24.3 / 24.9 / 31.0 / 31.9 / 32.6 | 31.0 |
| rig 16-31 + the conductor's own 0-15, ask then compute (before) | 65.5 / 73.4 / 74.5 / 76.4 / 82.7 | 74.5 |
| the same, the conductor computes while the rig answers (`--overlap`, S4) | 29.6 / 29.7 / 30.6 / 34.2 / 34.5 | 30.6 |
| the conductor alone, all 32 experts (reference) | 44.2-46.8 | 44.4 |

Concurrent asking halves the two-host layer (64.6 to 31.0 ms, 2.1x), overlapping the conductor's own work takes 74.5 to 30.6 ms (2.4x). The doc's estimate above says 136 ms to
84 ms (1.6x) for the whole four-host fleet; this toy has two hosts and no dense work, so the ratio is not comparable and nothing here predicts the real tokens a second. The
real number needs the real engine: not run.

**Real weights (Captain 1's go).** Nothing new to configure: the route list is the S3 plan's (`--remote-experts a:7840=0-39/0-127,b:7840=0-39/128-255,...`) and the engine now
overlaps by itself. To get the before-arm on the same setup, start the conductor once with `STRATA_NET_SERIAL=1` in its environment (one process, same flags); the
`strata net: decode: N layer calls, M with 2+ hosts in flight` line and `mean wait for the answers after the local CPU work` in the exit log are the per-layer numbers.
The CUDA `strata-dsv41` was compiled (object only) on the rig; it was not run: no real weights.

## The prompt through the servers (BBXL S5)

Captain 1's order (2026-11-03, #22829): send the prompt (prefill) through the expert servers too, so the conductor does not stream every expert from its own SSD.

**Two prompt paths exist in the engine, and only one of them reached the servers.** The WINDOW path (`feed_windows`, kWin = 16 tokens a window) already runs every served layer's
CPU lane through `cpu_experts` -> `net::Remote` (decode and short prompts). The CHUNKED flow (`forward_chunks`, long prompts: chunks of up to 16384 tokens, layer-major, experts
picked by the chunk streamed from the conductor's own files into VRAM slots) never asks `Remote`, which is why the status table said "the conductor needs the whole model on its
disk". `pick_path` chooses the flow from the prompt's length alone (`pick_flow_min`).

| Piece | Change |
| --- | --- |
| window size on the wire | the server accepts prompt windows of up to **32 tokens** (`kMaxPromptTok`; was 16 = `kMaxTok`, the engine's decode window, which is unchanged). 32 tokens at hidden 5120 / top-6 are a 655 KB request and a 3.9 MB answer, inside the wire's frame limits (1 MiB / 4 MiB); `Remote` lowers its window for a bigger hidden size. Raised because an expert picked by several tokens of a window is read once for all of them (the server computes up to 8 tokens of one expert per kernel call). An older server (limit 16) refuses a 32-token window with ERR: both ends of the prompt path need this build; `STRATA_DS4_PROMPT_WIN=16` on the conductor keeps the old size. |
| `net::Remote::prompt` (`ds4_net.cpp`) | T tokens of one layer (any T: the caller slices a long chunk into slabs), the S3 renumbering applied, windows of up to `prompt_window()` tokens with up to `STRATA_NET_PIPELINE` (default 4) in flight on every host (the Qwen B3/B5 `NetExperts::prompt`, now with a window parameter). Returns the expert rows and `got[i]`: entries no route owns (the conductor's own experts) and every entry after a failed request stay for the caller, who computes them from its own copy; the failed route is dropped and retried as in decode. A conductor missing a served layer entirely never needs the fallback unless a server dies. |
| `NetExperts::owns`, `prompt(..., max_tok)` | two small additions to `core` (the Qwen path passes nothing and is unchanged). |
| engine, opt-in `STRATA_DS4_NET_PREFILL=1` (only with `--remote-experts`) | `feed`, `feed_mark`, `feed_marks` take the window path instead of the flow, so no expert of a served layer is read from the conductor's disk. **Off by default**: a 16-token window pays a round trip a layer (no pipelining across windows: window n+1's layers need window n's attention state), so a long prompt is slower than the flow whenever the conductor does have the experts locally. Measure before turning it on (commands below). |

**Not done: the chunked flow itself does not call `Remote::prompt`.** That is the fast version (C tokens a layer, 32-token windows pipelined 4 deep on every host, no per-window
round trip). The hook is where the flow turns a chunk's picks for a SERVED layer into stream items (`flow_spec` / `flow_pump`, `run_dev_layers`): copy the chunk's FFN input and the
routed ids to the host, `Remote::prompt` in slabs, weight and sum the rows exactly as the local path's combine does, copy the partial sums back in stream order, and skip the layer's
streams and expert kernels. It is a change inside the async multi-stream GPU pipeline of `ds4_model.cu`; it needs a running engine with weights (the synthetic packs have no dense
part, the T340 has no CUDA), so it was not attempted blind. `Remote::prompt` is the whole host side of it and is tested.

**Tests (CI: `tools/dsv41/synthetic_split_test.py`, steps 3f-3j; failing first).** Against the base binaries every one fails (`unknown option --prompt-path`, `--slab`: 9 of the 17
failures that run reports; the other 8 are S4's checks); on this branch the whole script passes. `ds4_net_test --prompt-path` drives a prompt of any length through the layers in slabs
(`--slab`) with `Remote::prompt`, as the engine's chunk path would; the local reference computes the same rows on `ExpertHost` in windows. A 100-token prompt, 4 tokens, row hash
over every expert row:

| Check | Result |
| --- | --- |
| conductor pack = a slice holding layers 0-1 ONLY (`slice_stream.py emit pack 0-1`), layers 2-7 on one server | tokens and every row identical to the local run; 103 tokens x 6 x 6 = 3708 rows from the server, 0 computed here |
| control: that slice run alone, no server | does NOT reproduce the reference rows (the bytes really are missing) |
| server owns experts 16-31 only, the conductor its 0-15 | identical, some rows remote and some here |
| experts split over two servers, windows of 32 and of 16 (`STRATA_DS4_PROMPT_WIN`) | identical in both: a row does not depend on its window |
| a server dies in the middle of the prompt (`STRATA_DS4_SERVER_DIE_AFTER=12`) | identical through the local fallback |

**Measured: prefill tok/s at 2k and 16k tokens (synthetic, CPU kernels).** T340 conductor; the servers are the T340's own (loopback, experts 0-15, 4 threads) and the rig's (ssh tunnel,
experts 16-31, 5 threads). Pack: 2 layers x 32 experts, hidden 5120, ffn 512 (the real wire shape with cheap compute, so 16k tokens finish in a minute; the real ffn is 2304). Prompts in slabs
of 512 tokens, a 512-token warm-up (its own prefix) before each arm, then every timed run with its own prompt (`--prompt-salt`: unique prefix). The row hash of the 2k prompt is
`f7e72e76f25e1b50` for both the local arm and the servers (hash off in the timed runs). "Local" is this box's CPU doing the experts (the stand-in for the conductor's own copy; the
real local path is the flow streaming from an SSD into VRAM, which a synthetic pack cannot reproduce: the packs are 0.1 GB and sit in the page cache, so SATA against NVMe (S6) does not show here).

| Arm | 2048 tokens (tok/s, 3 runs) | 16384 tokens (tok/s, 2 runs) |
| --- | --- | --- |
| local, the conductor's CPU, 8 threads, windows of 32 | 459.9 / 461.2 / 495.3 | 461.7 / 494.4 |
| through the servers, windows of 32, 4 in flight (default) | 453.4 / 443.7 / 452.3 | 463.4 / 461.8 |
| through the servers, windows of 16 (the old limit) | 430.2 / 425.6 / 441.1 | 432.1 / 433.2 |
| through the servers, windows of 32, one at a time (`STRATA_NET_PIPELINE=1`) | 334.5 / 331.9 / 333.6 | 336.2 / 334.6 |

Read it for what it is: everything here is CPU-bound and the two servers' 9 threads about equal the local arm's 8, so "servers = local" says the wire (over a tunnel to the rig) costs
nothing once windows are pipelined, not that remote is faster. The levers show clearly: pipelining 4 windows deep is +36 % over one at a time, and the 32-token window +5 % over 16. At
16384 tokens the rate does not fall: the path is a steady stream, not a latency chain.

**Real weights (Captain 1's go; nothing was loaded).** Conductor with only its own layers on disk (`slice_stream.py emit PACK 0-K` for the conductor's layers), the same server routes as
S3/S4:

```sh
# A/B 1: the window path through the servers (needs both ends of this build; the conductor has no copy of the served layers)
STRATA_DS4_NET_PREFILL=1 strata-dsv41 --pack CONDUCTOR_PACK ... --remote-experts <the plan's line> --profile-run prompt-2k.ids   # then prompt-16k.ids, unique prefixes
# A/B 2: the flow from the conductor's own SSD (the full pack, no STRATA_DS4_NET_PREFILL): compare the "prompt N tok/s" lines of the two runs
# window size: STRATA_DS4_PROMPT_WIN=16 (the old limit); depth: STRATA_NET_PIPELINE=1..8
```

The exit log prints, per server, `prompt path: N windows, N rows, ... ms (server compute ... ms)` once `Remote::prompt` is used (the flow hook above), and
`decode: N layer calls, M with 2+ hosts in flight` for the window path.

## Storage tiers: experts on NVMe, never SATA (BBXL S6)

Captain 1's order (2026-11-03): every expert slice and conductor spill goes on NVMe, never SATA; `plan_split.py` takes a per-host storage tier (VRAM > RAM > NVMe > SATA)
and refuses SATA for experts when NVMe room exists; prefill reads from NVMe too. Survey of 2026-11-03 (gaming PC and stream PC from the order, the rig and the T340 read
only by this session with `lsblk` / `df` / `findmnt`; nothing was changed or deleted):

| Host | NVMe / RAM for experts | SATA (never for experts) | Note |
| --- | --- | --- | --- |
| gaming PC | 3 NVMe mounts: `Games` 30 GB free (after Flash-Next), `/mnt/ai` 38, `/mnt/ai_models` 36 = **104 GB** | `/mnt/bulk` (860 EVO, 550 MB/s; where the 2.00 set sits today) | RAM floor 12 GB free: 35 GB VRAM + 52 GB page cache of 64 leaves 12 |
| rig | WD Green SN3000 1 TB `/mnt/models`: **41 GB free** (96 % used) | `/` and `/home` are LVM on a PNY 500 GB **SATA** SSD: **`/var/tmp` is on that SATA disk, so today's slice there is on SATA**; 17 GB free | RAM floor 10 GB free (15 free with the fleet loaded). A stale NFS mount to the gaming PC (`/mnt/gaming_bulk_models`) makes a bare `df -h` hang: ask for the paths |
| T340 | none: RAM (tmpfs `/dev/shm`) only. 29 GB available, floor 16 GB free, so **13 GB** | `/` WD spinning disk behind the PERC, 11 GB free (98 %) | no NVMe, loads no AI models |
| stream PC | `C:` 500 GB NVMe, 16 GB free, floor 15 GB free: **1 GB usable** (a new SSD is coming) | `D:` SATA and failing: **excluded, never used** | RAM 64 GB, about 16 used by the stream |

**The schema.** A host in the `--hosts` JSON gets `"storage": [{"name", "kind": "tmpfs" | "nvme" | "sata" | "hdd" | "excluded", "free_gb", "reserve_gb", "path"}, ...]`;
`tools/dsv41/fleet.json` is the survey above in this form (hop order, the same memory, VRAM and per-call numbers as the built-in fleet, which is unchanged: a host without
`"storage"` keeps its single `disk_gb`, taken as NVMe, so every earlier plan prints exactly as before). A host's room for experts is
`min(its memory budget, its tmpfs + NVMe room)`; SATA adds nothing. Optional `"extra_gb"` is the conductor's spill (dense 6.2 GB + Engram 111 GB), reported beside the experts.

**The rule** (tests: `tools/dsv41/test_plan_split.py`, class `Storage`; each failed before the change):

- Experts go to tmpfs, then NVMe (the largest mount first), whole layers at a time: a layer's experts never straddle two disks, so the plan names the mount of every slice.
- SATA (and hdd) holds experts only with `--allow-sata`, and only when the tmpfs / NVMe of ALL hosts together cannot hold the set; then only the tail that did not fit (layer-major:
  the SATA disk of the host that holds the layers before it; popularity: after every NVMe tier, remote hosts only) and every such slice is printed `LAST RESORT: SATA`. If NVMe room
  exists but whole layers or expert ranges do not fit it, the run is refused and says so (use `--popularity` or more `--groups`), not rescued with SATA.
- An `excluded` mount (the stream PC's `D:`) is never used, with or without the flag.
- A refused run says how much SATA is on the hosts and that it was not used. A bad storage entry (unknown kind, negative size, no name) is refused with exit 2.
- Output (`--json`: `storage.targets`, `storage.sata_used_gb`): `slice rig -> /mnt/models (nvme) /mnt/models/strata/dsv41  layers 0-39/E254-290  12.7 GB`. That path is the
  `slice_stream.py apply` target. A host with two mounts gets one `apply` and one server per mount (`--serve-layers` of that line).

**Prefill and the conductor's spill.** The conductor needs its own experts on disk (the layers it computes, and the cold tail with `--allow-ssd`) and the dense and Engram files. They
must be on NVMe as well: Engram rows are random reads, where SATA is 5x slower on throughput and far worse on latency. `--conductor-copy all` plans for the whole set on the conductor
(what it needs until the prompt path runs through the servers, see Next). The planner does not move files; it prints the target paths for the existing tools.

**How much of the 2.00 bpw set (132.07 GB experts, 40 layers; the planner takes 3.30 GB a layer, the real layers are 2.97-3.93) fits on NVMe, and what to free**
(`plan_split.py --uniform-gb 3.30175 --hosts tools/dsv41/fleet.json --storage-report`; "experts" = what the host holds when disk is not the limit and its memory budget is, in the
hop order of the layer-major plan; a host with no NVMe keeps its tmpfs as the limit):

| Host | experts it holds | NVMe / tmpfs room | fits | free to hold its share | + conductor spill (dense 6.2 + Engram 111) |
| --- | ---: | ---: | --- | ---: | ---: |
| gaming PC | 85.8 GB | 104.0 | yes, 18 GB spare | 0 | 99.0 GB more, or 145.3 GB more with `--conductor-copy all` (the whole 132.07 GB set on the conductor) |
| rig | 9.9 | 41.0 | yes | 0 | - |
| T340 | 9.9 | 13.0 (tmpfs) | yes | 0 | - |
| stream PC | 26.4 | 1.0 | **no** | **25.4 GB on `C:`** (or the new SSD) | - |
| fleet | 132.1 | 159.0 | 106.7 GB of the set fits on NVMe / tmpfs as the disks are; **132.1 once 25.4 GB are freed on the stream PC** | 25.4 | 99.0 (gaming PC) |

Read off the table: today's plan with the survey's disks **refuses** (8 layers, 26.4 GB, have no host: `plan_split.py --uniform-gb 3.30175 --hosts tools/dsv41/fleet.json`, exit 3), and the
conductor's spill (dense + Engram = 117.2 GB) is the larger problem: it does not fit the gaming PC's 104 GB of NVMe beside 85.8 GB of experts. With the popularity plan (`--popularity`) the stream
PC's fast CPU draws 45.4 GB instead of 26.4 (free 44.4 GB there) and the rig and T340 draw nothing; give the stream PC a `reserve`/cap if it should not be the big host. The conductor's NVMe
needs `max(0, 85.8 + 117.2 - 104.0) = 99.0 GB` for experts plus spill. The rig needs nothing freed and should serve from `/mnt/models`, not `/var/tmp`.
RAM floors (gaming 12, T340 16, rig 10 GB free): the planner's memory numbers are page cache, which the kernel gives back; the rig's 13 GB leaves only 2 of the 15 free GB if those pages were
pinned, so do not `mlock` them.

**Cleanup candidates (text only, proposals for the cleanup list; nothing was posted or deleted).** I could read only the rig and the T340:

| Host / path | Size | Why a candidate | Verdict |
| --- | ---: | --- | --- |
| stream PC `C:` | need 25.4 GB | the slice's room | not surveyed (I did not touch that PC); a `C:` survey or the new SSD |
| gaming PC `/mnt/ai`, `/mnt/ai_models`, `Games` | need 99.0 GB for the spill (or 145.3 with the whole set) | the conductor's NVMe | not surveyed; or leave the Engram off the NVMe and say so |
| rig `/mnt/models/comfyui` | 499 GB (unet 188, diffusion_models 107, clip 62, text_encoders 58, checkpoints 44) | the rig does not need room for this set; only if the rig is to hold more than 13 GB | needs the owner's call |
| rig `/mnt/models/tabbyapi` | 161 GB (`gpt-oss-120b-4.0bpw-exl3` 36, two `Qwen3.8-27B` 4.0 / 5.0 bpw builds with DFlash variants 19 + 16 + 16, `qwen14b_5bpw` 10) | duplicates of one model at several bit widths | needs the owner's call |
| rig `/mnt/models/{bb_models 61, hf_train 23, muse-glimmer 19, qwen38-uncensored 24}` | 127 GB | BigBrain's Devstral-2-123B (the engine's own models: keep), a training merge, a 30B GGUF | needs the owner's call |
| rig `/mnt/models/swapfile-ai` | 30 GB | active swap (21.9 GB in use): not a candidate | keep |

## GPU expert server (`--expert-server-gpu`, BBXL)

`strata-dsv41 --expert-server-gpu` (or the engine-less `strata-dsv41-expert-server-gpu`) keeps the served layers' routed experts in VRAM and
answers a window with the engine's own kernels: the window's (token, expert) pairs become a plan of blob addresses, `moe_win_group` groups them
by expert (each expert read once for all the tokens that picked it), `gguf::win_rows` (`win_experts` without the combine) runs gate / up, the
clamped swiglu and down on the card, and the un-weighted rows go back over the same SXP1 wire. Code: `ds4_net_gpu.cu` (upload, compute),
`ds4_gpu_place.hpp` (fit rules, no CUDA, unit-tested by `ds4_gpu_place_test`, which CI's ctest runs). Flags (beside `--pack --listen --serve-layers --serve-experts`):

| Flag | Meaning |
| --- | --- |
| `--expert-server-gpu` | experts in VRAM; implies `--expert-server` |
| `--gpu-vram-gb G` | most VRAM the experts may take (default: free VRAM minus the reserve) |
| `--gpu-reserve-mib N` | kept free after the experts and the work buffers (default 256) |
| `--gpu-device N` | CUDA ordinal (default 0; pin a rig card with `CUDA_VISIBLE_DEVICES=<uuid>`, never by index) |
| `--gpu-cpu-rest` | layers that do not fit, or whose expert type has no device kernel, are served by the CPU lane |

Without `--gpu-cpu-rest` a server that cannot hold everything refuses to start and prints the shortfall (1.58 GiB of experts, 0.5 GiB cap:
"short by 1.08 GiB"). The CPU-only binary answers `--expert-server-gpu` with "this build has no CUDA". The server sets `DS4_IQ_Q8K=1` unless the
environment says otherwise: the device rounds a low-bit expert's activations per 32 values by default, the CPU lane per 256 (ggml's Q8_K);
without it the rows differ by 10-30 % of their RMS on random weights, with it by about 1e-6 (MXFP4 and most layers) to 1e-3.

**Rows are not bit-identical to the CPU server's** (as for Qwen's `--expert-server-gpu`): the two sum a block's products in another order, and an
activation on an int8 rounding boundary can flip one quantum. `ds4_gpu_rows_test` (CUDA, built by name) compares both hosts on windows of 1-16
tokens with shared experts and masked entries: unasked rows untouched, every row finite and within 5e-2 of its RMS, at most 1 % of rows beyond 2e-3. A
conductor that falls back to its own CPU lane mixes the two kinds of row; greedy tokens agree except near ties (the random toy below diverges at
token 5, deterministically: the same GPU server twice gives the same hash).

### Measured (2026-11-03, synthetic pack of the real shape, no real weights)

Rig, RTX 3090 (sm_86, CUDA 12.4, gcc 13) pinned by UUID, TabbyAPI's 1.9 GB on it idle; CPU server = the T340 binary on the rig's i5-9400, 6 threads.
Pack: hidden 5120, FFN 2304, top-6, 4 layers x 48 experts (IQ1_M, IQ2_XXS / IQ1_M, IQ2_XS / IQ1_M, IQ2_XXS / IQ2_XS), 1.58 GiB in VRAM
(upload 2.2 s warm, 19 s cold). A layer call, 200 calls a window size after 20 warm-up, fresh random picks and layer every call (`ds4_gpu_rows_test --bench 200`):

| window | CPU server ms | GPU server ms | GPU speed-up | CPU vs 1 token | GPU vs 1 token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.114 | 0.327 | 12.6x | 1.00 | 1.00 |
| 2 | 8.429 | 0.635 | 13.3x | 2.05 | 1.94 |
| 4 (a K=4 verify window) | 12.450 | 1.140 | 10.9x | 3.03 | 3.48 |
| 8 | 21.170 | 2.153 | 9.8x | 5.15 | 6.58 |
| 16 | 34.932 | 4.276 | 8.2x | 8.49 | 13.07 |

**The K=4 verify window costs 3.5x one token on the GPU server and 3.0x on this rig's CPU (S1 measured 4.8x on the T340): the GPU is not flat in the window.**
Distinct experts are read once, but a window of 4 touches about 20 experts against 6 for one token, and a 3090 reading 8.6 MB experts is
memory bound, so compute grows with the distinct experts. What changes is its size next to the round trip: the server's own mean compute a call over the 64-token run was
0.416 ms (GPU) against 4.516 ms (CPU). With a fixed 1 ms of wire and protocol a call (an estimate from the decode below, loopback), a K=4 window costs about 1.7x one token on the
GPU server and 2.6x on the CPU one: DSpark pays better through the GPU server, but a window is never nearly free. Not measured: a K=4 window through the wire.

Conductor's split test (`ds4_net_test --mode remote`, 64 greedy tokens: 63 single-token steps x 4 layers = 252 calls, the toy's dense part included):

| Server | Conductor | ms a token | tok/s |
| --- | --- | ---: | ---: |
| CPU (rig) | on the rig, loopback | 20.5 | 48.9 |
| GPU (rig 3090) | on the rig, loopback | 4.8 | 207 |
| CPU (rig) | T340 through an ssh tunnel (the rig's ufw does not admit the T340 on 7841-7842 directly; left as it is) | 27.4 | 36.6 |
| GPU (rig 3090) | T340 through an ssh tunnel | 14.0 | 71.5 |

CPU-server rows hash `e7f2d1b3164e44c1` as in the table above; the GPU server's is `145113bd71350179` every run. The tunnel adds about 2.2 ms a call, and the T340 runs the toy's dense part slower than the rig.
Not done: the 2k / 16k prompts with unique prefixes. An expert server's cost does not depend on the context length (attention is the conductor's), and the prompt path
through the servers (Next, item 3) and the real weights are not there, so no such run exists; the numbers above are per layer call and per toy token.

### Commands

```sh
# rig (Ubuntu, CUDA 12.4: nvcc wants gcc 13 as host compiler; sm_86 for the 3090 / 3060s; the rig's 5070 is sm_120 and needs CUDA 12.8+)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DCMAKE_CUDA_ARCHITECTURES=86-real \
      -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build --target strata-dsv41-expert-server-gpu ds4_gpu_rows_test -j3     # 37 steps once the libraries are built
CUDA_VISIBLE_DEVICES=<card uuid> STRATA_EXPERT_TOKEN=... build/strata-dsv41-expert-server-gpu --pack PACK --listen 0.0.0.0:7841 \
      --expert-server-gpu --serve-layers 27-29 [--gpu-vram-gb 20] [--gpu-cpu-rest]
CUDA_VISIBLE_DEVICES=<card uuid> build/ds4_gpu_rows_test --pack SYNTHETIC_PACK [--bench 200]     # rows vs the CPU server; needs a free card
```

The rig has six cards, pinned by UUID (`add_a_3060.md`): the 3090 (24 GB) holds seven 3.3 GB layers, each 3060 (12 GB) about three. The three 3060s and the 5070 hold
TabbyAPI models today, so a real run needs them unloaded, as the Placement table says. **Stream PC (RTX 5070 Ti, sm_120):** the same two targets with
`-DCMAKE_CUDA_ARCHITECTURES=120-real` and CUDA 13.x (the build section above, TODO-1276); `strata-dsv41-expert-server-gpu.exe` takes the same flags, and
`--gpu-vram-gb 5` fits what the Placement table says is free beside TabbyAPI. Not built or run there (no toolkit; the stream PC is not touched from this work).

**Prompt windows on the GPU server (S3 x S5 merge).** The server accepts prompt windows of up to 32 tokens (`kMaxPromptTok`), but the GPU lane's buffers and the engine's window kernels hold 16 (`kMaxTok`):
a window longer than 16 tokens is computed on the card in sub-windows of 16 (token-major rows, so each piece is an offset; same rows as one window). Not compiled here (no CUDA / C++ toolchain on this PC): CI builds the CPU side; the CUDA side needs a GPU build.

## Faults, security

As for Qwen (docs/REMOTE_EXPERTS.md): shared token, the server binds loopback unless told otherwise, a route that fails is computed locally and retried
with backoff, a request outside the served layers or experts is refused, one conductor at a time per server. A conductor of the other engine family is refused
at the handshake. The stream PC's server is a Task Scheduler one-shot (never started from an agent shell); stream mode stops it by its PID.
Cross-OS rows were identical in the test above; the swiglu `exp` is the host's libm, so a different libm could differ in the last bit of a row (greedy tokens
would still agree except on near-ties) - the synthetic test and `ROWHASH` are how to see it.

## Next

1. The real-weights split (RESULTS.md gate): conductor `strata-dsv41` on the gaming PC, T340 + stream PC (CPU servers) with slices of their layers, 64 tokens against the single-process
   reference, tok/s and round trip. Until then the synthetic test and the row-hash runs above are the evidence.
2. The GPU expert server on the stream PC's 5070 Ti and on real weights (done on the rig's 3090 with a synthetic pack: "GPU expert server" above).
3. The chunked flow's hook for `Remote::prompt` (BBXL S5 "Not done"): the host side is done; the engine side needs a run with real weights.
4. Disk: the 3 bit set needs about 60 GB more on the rig and 25 GB on the T340, or a NAS mount.
