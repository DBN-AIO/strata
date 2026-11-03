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

The server needs the engine's CPU expert kernels and a CUDA runtime for the dense weights. A CUDA-free server build is
the next step: the dispatch already runs every row on the CPU when `host_res == nullptr` and `plan == nullptr`, so only
the engine's start-up (dense weights, PLE, cache) has to be skipped in `--expert-server` mode.

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

## Expert servers on GPU (2026-11-03)

An expert server can hold its served layers' experts in VRAM and compute them with the GPU kernels instead of the CPU.
It uses the helper GPU tiers that already exist (`--expert-cache-device1..3`, `RemoteExperts`, docs/SECOND_GPU.md):
in `--expert-server` mode

- the helpers are filled with the served layers only (profile order first, then every other served expert), not with
  pairs from all 48 layers;
- CUDA0's own cache does not keep an expert out of the helpers (in server mode no verify window runs on CUDA0, so an
  expert it held would only fall back to the CPU); CUDA0 holds the dense weights, `--expert-cache 16` is enough;
- the RAM warm-up touches only the experts no helper holds;
- the server logs how many served experts sit in helper VRAM, and at each disconnect how many requested rows the GPUs
  computed (`... 6366 rows asked, 6317 on GPU (99.2%)`).

Entries a helper does not hold stay on the server's CPU, as before. The GPU kernels round differently from the CPU
kernels (the same as the conductor's own cache, bench/results/2026-09-27-cache-parity), so a GPU server's rows are not
bit-identical to a CPU server's and the generated text can differ.

Rig command (GPUs pinned by UUID; CUDA0 = the RTX 3090 for the dense weights, CUDA1..3 = the three RTX 3060):

```sh
strata --pack PACK --native SHARD1.gguf --max-context 1024 --prefill auto --spec 2 \
       --expert-profile data/expert-profile.bin --mmap-experts --pcie-frac 0 \
       --expert-server --listen 7840 --serve-layers 38-47 --expert-cache 16 \
       --gpu GPU-d526893c-...,GPU-1608aa48-...,GPU-4382b370-...,GPU-1bab2dba-... \
       --expert-cache-device1 N1 --expert-cache-device2 N2 --expert-cache-device3 N3
```

Size each `N` from the card's live free memory: while TabbyAPI runs, a card may only take its free memory minus 1 GB,
and the 3090 at most 6 GB in total. With about 1.36 MB per blob slot plus about 400 MiB of context and scratch,
`N = (free_MiB - 1024 - 400) / 1.36`. The rig script that does this and refuses to start when the 3090 lacks room is
in the B2 PR description. Measured on the rig: CUDA0 (3090) +3924 MiB, each 3060 +2135..2206 MiB, all under their caps.
The RTX 5070 (sm_120) is not used: the binary is built for sm_86, and the engine takes at most three helper devices.

### Measured (B2)

Same as above: Qwen3.8-Flash-Next ISTA Q2_0, 14 prompt tokens, 64 new tokens, greedy, `--spec 2`, `--pcie-frac 0`,
layers 38-47 served by the rig over the SSH tunnel. Conductor on the gaming PC with `--expert-cache 1100` (5662 MiB of
VRAM; the 1500 slots of the first measurement peaked at 6172 MiB, over the 6 GB cap). Rig server binary with this
change, CPU arm = the same binary without helpers. Runs alternate (GPU, CPU, GPU, CPU), page cache warm.

| Server | decode tok/s | calls | round trip | server compute | rows on GPU | tokens per round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU (i5-9400, AVX2), run 1 | 17.21 | 420 | 4.522 ms | 1.240 ms | 0 % | 1.52 |
| CPU, run 2 | 16.35 | 410 | 4.810 ms | 1.274 ms | 0 % | 1.56 |
| GPU (3 x RTX 3060, 4485 of 5120 experts, 5.77 GiB), run 1 | 17.82 | 600 | 2.609 ms | 0.442 ms | 99.2 % | 1.07 |
| GPU, run 2 | 11.96 | 600 | 2.594 ms | 0.425 ms | 99.2 % | 1.07 |

- Per layer call the GPU server is 2.8x faster to compute (0.43 against 1.26 ms) and the round trip falls 44 %
  (2.60 against 4.67 ms). The remaining 2.2 ms is the SSH tunnel and the frame copies, not compute.
- Decode tok/s is not a clean measure of the server here: the conductor's own work dominates (its CPU pool and file
  reads), and the GPU server's different rounding changed the generated text, so its drafts were accepted less often
  (1.07 against 1.5 tokens per round, 60 against 42 rounds). Remote time per decode round: CPU 45 ms, GPU 26 ms.
- Against the 2.92 tok/s of the first rig measurement: that run had 14.3 ms of rig CPU compute per call with TabbyAPI
  busy beside it; today the rig CPU took 1.24 ms, so the gap the GPU closes is smaller than it was then. The GPU path
  does not depend on how busy the rig's 6 CPU cores are.
- Two CPU runs gave different texts: the conductor's adaptive tier (swaps every 4 rounds, without waiting) moves experts
  between its GPU and CPU on timing, so runs differ slightly. `--adapt-every 0` (a static cache) makes runs repeatable.

## NVLink

The two RTX 3090 of the gaming PC are NVLink-bridged (`nvidia-smi topo -m`: NV4, four links, 14 GB/s per link shown).
`cudaDeviceCanAccessPeer` is 1 in both directions, `cudaDeviceEnablePeerAccess` succeeds, `cudaDevP2PAttrAccessSupported`
and `NativeAtomicSupported` are 1. The fork's layer split / split engine does not use it: on NVIDIA it exchanges sums
through pinned host memory and calls no peer API (docs/MULTI_GPU.md "No NVLink or peer-to-peer access is needed",
docs/SPLIT_ENGINE.md). Using NVLink for the stage-to-stage hidden state would be a change in the split exchange, not in
the remote-expert path.

## Next steps

1. Route the prompt path (prefill) through the expert servers, so the conductor needs no local copy of handed-out layers.
2. Done for CUDA helper GPUs ("Expert servers on GPU" above); still open: the stream PC's 5070 Ti, more than three helper cards. Was: load the served layers' experts into VRAM and
   compute them with the GPU kernels, so the host CPU stops being the bottleneck.
3. Overlap: send layer l+1's request as soon as its router output exists (today one round trip per layer, in series).
4. CPU-only server build for the T340 (no CUDA at start-up), fp16 hidden states on the wire.
