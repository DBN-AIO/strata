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
3. The prompt path through the servers (as B3 did for Qwen), so the conductor needs no local copy of served layers.
4. Disk: the 3 bit set needs about 60 GB more on the rig and 25 GB on the T340, or a NAS mount.
