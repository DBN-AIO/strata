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
| GPU expert server for dsv41 (rig 72 GB VRAM, stream 5070 Ti) | not done; needs the engine's VRAM expert slots behind `serve_experts` (see "Next") |
| parallel remote calls, conductor work overlapped with the servers (BBXL S4) | done, measured on synthetic packs (T340 + rig): "Parallel remote calls (BBXL S4)"; the real engine has not run it |
| prompt path through the servers (BBXL S5) | host side done and tested (`Remote::prompt`, 32-token windows, a conductor pack without the served layers passes the synthetic test); engine: opt-in `STRATA_DS4_NET_PREFILL=1` sends the prompt down the window path (compiled, not run). **Not done:** the chunked flow itself does not call `Remote::prompt` (needs a running engine). "The prompt through the servers (BBXL S5)" |
| chunked prompt flow through the servers + the combined real-weights script (BBXL S7, stacked on S5) | host side done and tested (`Remote::owns`, `ds4_net_chunk.hpp`, synthetic steps 3l-3n); engine: `STRATA_DS4_NET_PREFILL=2` makes the chunked flow ask the servers for a served layer's experts (**compiled on the rig, not run**: no real weights); `tools/dsv41/bbxl_realweights_run.py` runs S1-S7 before/after arms in one command (dry-run and synthetic end to end tested, **not run on real weights**). "The chunked flow through the servers (BBXL S7)", "The combined real-weights run (BBXL S7)" |

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

**(Done in S7, see "The chunked flow through the servers (BBXL S7)" below.) The chunked flow itself did not call `Remote::prompt` in S5.** That is the fast version (C tokens a layer, 32-token windows pipelined 4 deep on every host, no per-window
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

## The chunked flow through the servers (BBXL S7)

Captain 1's order (2026-11-03, #22865 item 1): finish the fast chunked prompt flow (`forward_chunks` / `flow_pump`) calling `Remote::prompt`. Stacked on S5 (`max2/s5-prefill-net`).

**What the engine does with `STRATA_DS4_NET_PREFILL=2`** (needs `--remote-experts`; `=1` is still S5's window path; unset = the flow reads every expert from this PC's files, as before):

| Piece | Change |
| --- | --- |
| which experts | `fl_owned(layer, e)`: the expert is in the CPU lane's list (not a VRAM slot) and a route owns it (`Remote::owns`, the S3 renumbering applied). Only for a served layer, not the draft's (`mtp`), not a tensor-parallel group (`nm > 1`). |
| the stream | `flow_spec` / `flow_rest` push NO flow item for an owned expert (no file read, no staging or ring slot, no DMA, no expert kernel), and give owned experts the LAST virtual slots of the layer, so the streamed ones stay one run of slots (the batches take their bounds from the first to the last item: a skipped expert with rows in between would be computed with the wrong weights). The expert cache never promotes an owned expert (`cpu_hits`): its bytes may not be on this PC. |
| the call | `net_chunk_layer` (called by `flow_layer` once the layer routed, while the local items are read): the chunk's routing (`c.ridx`, `c.rw`) to the host, then slabs of `STRATA_DS4_NET_SLAB` (512) tokens: the slab's FFN input rows to the host, `Remote::prompt` (windows of 32 tokens, 4 in flight per host), the rows to VRAM scratch (kept per chunk, 63 MB at 512 tokens, hidden 5120, top-6), and the engine's own `moe_combine` (acc = fma(w, row, acc), a token's picks in ascending order) adds them into the layer's accumulator. The conductor's VRAM and streamed experts go the flow's way as before. One `Remote` for all devices: their threads take a mutex around the call. |
| pure host part | `ds4_net_chunk.hpp` builds the slab (which picks are sent, where each row sits in the answer, the weights) and holds a CPU copy of `moe_combine`; the engine and `ds4_net_test --chunk-flow` run the same code. |
| no fallback | every owned pick must come back: a conductor pack without the served layers' bytes cannot compute it. A failed request ends the run with the error (the window path of S5 and decode still fall back when the conductor has the bytes). Without the expert flow (`DS4_FLOW=0`, or experts that are not a GGUF's in-place ones) a served layer is an error, not a silent read. |

A conductor pack WITHOUT the served layers' bytes: the flow has nothing to read for them. `STRATA_DS4_NET_PREFILL=2` and a pack made by `slice_stream.py emit PACK <the conductor's layers>`; the layers' tensors the pack still lists but whose experts are missing are never opened for a served layer by this path. **Not verified on the engine**: whether the engine's start-up (expert RAM tier, profile warm-up) touches a missing expert; the first real run answers it (the log's first error names the layer).

**Rounding.** The rows are bit-identical (the server's `ExpertHost`); the layer's sum adds the servers' contribution with `moe_combine` before the conductor's local experts instead of in slot order, so a prompt through servers can round differently from a prompt computed wholly here (as the window and flow paths already differ from each other, `pick_flow_min`). It is deterministic: the same prompt gives the same bits on every run.

**Tests (CI: `tools/dsv41/synthetic_split_test.py`, steps 3l-3n; failing first).** Against the S5 binaries 5 of the script's checks fail (`unknown option --chunk-own`, `--chunk-flow`); on this branch the whole script passes (46 checks). `ds4_net_test --chunk-flow` is the engine's hook on the toy (the code of `ds4_net_chunk.hpp`: only owned picks are sent, every one must come back, rows summed with `moe_combine`'s arithmetic) and prints `CHUNKSUM`, a hash of those partial sums; `--chunk-own LO-HI[:L0-L1]` in `--mode local` stands for the servers' share:

| Check | Result |
| --- | --- |
| server owns experts 16-31, the conductor its 0-15 | tokens, every row (`ROWHASH`) and every partial sum (`CHUNKSUM`, 2397 picks sent) identical to the one-process reference |
| conductor pack = layers 0-1 only, layers 2-7 on a server | identical; 103 tokens x 6 x 6 = 3708 rows from the server, 0 computed here for those layers |
| server dies mid-prompt | the run stops with an error naming the layer (no zeros, no silent fallback) |

**Measured (synthetic, the toy's CPU stand-in for the chunk flow).** T340 conductor (8 threads), T340 server (experts 0-15) + rig CPU server (experts 16-31, ssh tunnel), hidden 5120 / ffn 512, 2 layers, warm-up run first, unique prefixes, `ROWHASH` and `CHUNKSUM` equal local vs servers at 2k. The T340 was loaded by other work during the run (load average 6 on 4 cores), so absolute numbers are lower than S5's quiet run and the ratios are the result. Prefill tok/s:

| prompt | chunked local (conductor's CPU) | through servers, 32-token windows 4 in flight | through servers, one window at a time |
| --- | --- | --- | --- |
| 2k | 355 346 343 | 422 409 350 | 273 286 299 |
| 16k | 297 329 | 399 422 | 298 297 |

The toy is CPU-bound and has no dense part: it shows that the pipelined slab path is +20 % over the conductor computing everything itself here and +40 % over one window at a time, and that the bits are the same. It does NOT say that servers beat a local NVMe flow on the real model: that needs the real engine (below). The rig 3090 GPU server arm was not run: the GPU server (S2) and the 32-token prompt window (S5) are on different branches, so no binary has both until they merge (the rig 3090 itself has 22 GB free: `nvidia-smi`, GPU-d526893c).

**Not done.** The engine itself has not run: no real weights, and the synthetic packs have no dense part. `ds4_model.cu` compiled (object, CUDA 12.4, sm_86) on the rig. `net_chunk_layer` copies the slab through pageable host memory and does not overlap the next slab's call with the GPU combine (a measured first real run decides whether that is worth doing). `STRATA_DS4_NET_PREFILL=2` is off by default.

**Real weights (Captain 1's go): the one command in the next section** runs the S7 arm (`s7`) against the S5 arm and the flow from the conductor's own SSD (`base`) at 2k and 16k, unique prefixes, warm-up first.

## The combined real-weights run (BBXL S7)

Captain 1's order (#22865 item 1): one command that runs S1-S6 on real weights the moment the gaming PC is back: warm-up first, RESULTS table and dashboard rows written.

```sh
# on the conductor (the gaming PC). Hosts: tools/dsv41/fleet.json (BBXL S6). Site paths: copy tools/dsv41/bbxl_run_manifest.example.json and fill every <...>
python3 tools/dsv41/bbxl_realweights_run.py --manifest run.json --dry-run     # prints every command (preflight, servers, engine runs), runs none
python3 tools/dsv41/bbxl_realweights_run.py --manifest run.json               # the run: runs/bbxl-<time>/RESULTS.md + results.jsonl + every log
```

| Step | What |
| --- | --- |
| preflight | the probes run first and the script REFUSES (exit 3, nothing started) when a floor breaks: free RAM (`MemAvailable`) gaming >= 12 GB, T340 >= 16, rig >= 10; stream PC: C: free disk >= 15 GB; and every GPU the manifest names (by UUID, `nvidia-smi`) must have its `min_free_mib` (the rig 3090, GPU-d526893c-b378-4be2-4e99-71848237f300: 3000). Floors can be overridden per host in the manifest (`floors`). |
| arms | `base` = everything off (CPU servers, plan without popularity, `STRATA_NET_SERIAL=1`, `--dspark none`, prompt through the conductor's own flow). One change each: `s1` `--dspark auto`; `s2` GPU expert servers (`--expert-server-gpu`, a host without a GPU server in the manifest keeps its CPU server); `s3` the popularity plan + `STRATA_DS4_EXPERT_PERM` + `--expert-profile`; `s4` parallel remote calls (serial hook off); `s5` `STRATA_DS4_NET_PREFILL=1`; `s7` `STRATA_DS4_NET_PREFILL=2`; and `all`. Arms are grouped by (server kind, plan) so servers restart as seldom as possible; an arm the manifest cannot run (no `after` plan, no GPU server) is listed under "Not run". |
| runs | per arm: one warm-up run (not counted), then `--runs` (default 3) timed runs at 2k and 16k tokens, each with its OWN 16-token unique prefix (the same prefixes in every arm, so tokens can be compared). Metrics from the engine's own lines: `prompt: N tokens in X ms (Y tok/s` and `decode: N tokens in X ms (Z tok/s`. |
| servers | started and stopped by the script over ssh (`ssh` in the manifest's host entry; a host without it, or `"managed": false`, is only probed on its port); the token comes from a token FILE on each host (`token_file`), never from the command line; stopped on every exit, also an error. |
| outputs | `RESULTS.md`: the preflight table and one table row per arm and prompt (median, every run, ratio against `base`, whether the tokens equal `base`'s) ready to paste here; `results.jsonl`: the dashboard rows. |

Dashboard rows (no earlier BBXL row format existed in this repo; defined here, `schema` versioned). One JSON object per line, per arm x prompt x metric:

```json
{"schema": "bbxl-bench/1", "ts": "2026-11-04T09:00:00", "project": "bbxl", "engine": "strata-dsv41", "stage": "S7", "arm": "s7", "arm_what": "base + the chunked flow ...",
 "prompt": "16k", "prompt_tokens": 16384, "metric": "prefill", "value": 123.4, "unit": "tok/s", "runs": [120.1, 123.4, 125.0], "decode_tokens": 128, "tokens_match_base": true}
```

`metric` is `prefill` or `decode` (tok/s); `value` is the median of `runs`; `tokens_match_base` is null for `base` itself. GPU server rows (`s2`, `all`) are not bit-identical to the CPU server's (S2), so `tokens_match_base` can honestly be false there.

**Tests** (`tools/dsv41/test_bbxl_realweights_run.py`, in CI): one change per arm, the floors, the preflight refusal, `--dry-run` runs nothing, the RESULTS / JSON writer, the engine-line parsers; with a build dir the whole run end to end on a synthetic pack (`--synthetic BUILD_DIR PACK`: `ds4_net_test` as the engine, two loopback expert servers; arms `base`, `s4`, `s5`, `s7` run, the arms that need real weights or a GPU are listed as not run). The synthetic numbers prove the machinery, not the speed.

**Not done.** Never run on real weights, the stream PC's server is not managed by the script (its Task Scheduler one-shot stays manual), the ssh start path was exercised only as printed commands (`--dry-run`) and once live on the T340 -> rig leg for a harmless command.

## Faults, security

As for Qwen (docs/REMOTE_EXPERTS.md): shared token, the server binds loopback unless told otherwise, a route that fails is computed locally and retried
with backoff, a request outside the served layers or experts is refused, one conductor at a time per server. A conductor of the other engine family is refused
at the handshake. The stream PC's server is a Task Scheduler one-shot (never started from an agent shell); stream mode stops it by its PID.
Cross-OS rows were identical in the test above; the swiglu `exp` is the host's libm, so a different libm could differ in the last bit of a row (greedy tokens
would still agree except on near-ties) - the synthetic test and `ROWHASH` are how to see it.

## Next

1. The real-weights split (RESULTS.md gate): conductor `strata-dsv41` on the gaming PC, T340 + stream PC (CPU servers) with slices of their layers, 64 tokens against the single-process
   reference, tok/s and round trip. Until then the synthetic test and the row-hash runs above are the evidence.
2. A dsv41 GPU expert server (the engine's VRAM expert slots behind `serve_experts`, as `--expert-server-gpu` does for Qwen) for the rig and the stream PC's 5070 Ti.
3. The chunked flow's hook for `Remote::prompt` is written (BBXL S7, compiled, not run): the first real run is `bbxl_realweights_run.py` ("The combined real-weights run (BBXL S7)").
4. Disk: the 3 bit set needs about 60 GB more on the rig and 25 GB on the T340, or a NAS mount.
