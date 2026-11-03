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
