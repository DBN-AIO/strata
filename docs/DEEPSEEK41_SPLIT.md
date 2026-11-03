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
| prompt path through the servers | not done: the chunked prompt path streams experts from the conductor's own files, so the conductor needs the whole model on its disk |

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
3. The prompt path through the servers (as B3 did for Qwen), so the conductor needs no local copy of served layers.
4. Disk: the 3 bit set needs about 60 GB more on the rig and 25 GB on the T340, or a NAS mount.
