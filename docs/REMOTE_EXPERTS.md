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
4. CPU-only server build for the T340 (no CUDA at start-up), fp16 hidden states on the wire.
