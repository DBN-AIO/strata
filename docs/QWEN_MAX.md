# Qwen3.8-Max (Qwen3.8-2.4T-A95B) on Strata: the shapes, the split, the placement

Status: engine support proven on a Max-SHAPED synthetic model (below); **nothing here has run the real 400-660 GB weights** - none
were downloaded (only GGUF headers, by HTTP range request). Branch `qwen-max-shapes` of DBN-AIO/strata, 2026-11-03.

## 1. The geometry, read from the GGUF headers

`python tools/gguf_geometry.py --hf unsloth/Qwen3.8-2.4T-A95B-GGUF/UD-IQ1_M` fetches the first 4 MiB of every shard (64 MiB at most for
shard 1, whose 10.9 MB are the tokenizer and metadata and no tensor) and prints this. It never reads a weight; a tensor's size is its
block geometry, or the distance to the next tensor's offset for a type this tree has no kernel for. The tool is generic (any GGUF).

| | Qwen3.8-2.4T-A95B | key |
| --- | --- | --- |
| architecture | `qwen35moe` - the GGUF architecture of Qwen3.6-35B-A3B, not a new one | `general.architecture` |
| blocks | **93 = 92 trunk + 1 MTP** (`blk.92` carries `nextn.*`); trunk = 69 GDN + 23 full attention (every 4th) | `block_count`, `nextn_predict_layers`, `full_attention_interval` |
| hidden | 8192; vocabulary 248320; context 262144 | `embedding_length` |
| attention | 64 query heads, 4 KV heads, head width 256, rope 64 (sections 11,11,10,0, base 1e7) | `attention.*`, `rope.*` |
| GDN | conv 4, state 128, 16 key heads, **128 value heads**, inner 16384 (conv channels 20480) | `ssm.*` |
| MoE | **512 experts per block, top-10** + 1 shared; routed width 2048, shared width 2048 | `expert_count`, `expert_used_count`, `expert_*feed_forward_length` |
| n-gram / PLE tables | none | (no such tensors) |
| MTP | the in-file block 92: attention + MoE, experts in Q2_K, `eh_proj` Q8_0 | `blk.92.*` |

Bytes (exact, from the headers; "GB" = 10^9):

| file set | shards | bytes in the files | trunk experts' type | **bytes per expert** (gate+up+down) | experts per block | all experts (93 blocks) |
| --- | --- | --- | --- | --- | --- | --- |
| UD-Q1_0 | 10 | 397,256,393,248 | type 66 (1.1875 bpw) | **7,471,104** | 3.83 GB | 360.4 GB |
| UD-IQ1_M | 13 | 563,954,811,904 | IQ1_M (1.75 bpw) | **11,010,048** | 5.64 GB | 527.1 GB |
| UD-IQ2_XXS | 15 | 656,565,044,416 | IQ2_XXS (2.0625 bpw) | **12,976,128** | 6.64 GB | 619.7 GB |

All three: the MTP block's experts are Q2_K (16,515,072 B per expert, 8.46 GB for the block); attention/GDN/shared-expert matrices Q5_K,
`ssm_out` and the shared down Q6_K, `eh_proj` Q8_0, `token_embd` and `output` Q4_K (1.14 GB each), router, norms, `ssm_alpha/beta/conv` F32.
Dense (non-expert) bytes: 380,143,104 per GDN block, 342,067,200 per attention block, 34.6 GB for the 93 blocks, 36.9 GB with the
embedding and head. Same in all three sets.

## 2. What was hard-coded, and what reads the file now

Two engine paths exist. **Flash-Next** (`qwen4exp`, `src/core`, `src/program`) is compiled for its shape: `core::kCompiledGeometry`
(hidden 2560, 48 layers, FF 640, hyper-connections, indexer) is `static_assert`ed in about twenty kernels, and a file of another shape
is refused at load with the key that differs (`compiled_geometry_mismatch`). That is deliberate and unchanged. **Max runs on the other
path**, `src/qwen35` (`strata-q36`, the `qwen35moe` family), which already took its dimensions from the metadata
(`core::geometry_from_gguf`) and checks the file against a layout table (`check_gguf_layout`); its only compiled-in numbers are
head width 256, GDN state 128, conv kernel 4, at most 512 experts and top-k at most 15, all of which Max meets exactly. The allowlist of
architectures is `model_family_from_arch`: `qwen4exp` and `qwen35moe`, anything else is refused by name.

What still assumed Flash-Next's (or Qwen3.6's) shapes on the way to Max, found by running a Max-shaped model, and fixed:

| where | assumed | now |
| --- | --- | --- |
| `kernels/cpu/native_expert.hpp` `kNativeActBytes` / `kNativeHBytes` | activation of 2560 values (4096 B) and down activation of 640 values (1024 B) | 12288 B / 3072 B (Max needs 9344 / 2336); `native_fmt` refuses a layer that needs more and says the numbers |
| `kernels/cpu/pool.hpp` `SplitBufMulti::ff[MAXT][FF]` | FF = 640 floats per token (Flash-Next); Qwen3.6's 512 fit by luck | `kNativeMaxFF` = 2560; `native_fmt` refuses a wider expert (Max: 2048) |
| `kernels/cuda/iq_kernels.cu` `STRATA_D_FMTS` | a down projection is never IQ2_XXS / IQ1_M (Flash-Next: IQ4_NL / IQ4_XS / Q2_0; Qwen3.6: K-quants) | IQ2_XXS (16) and IQ1_M (29) take the down role, so UD-IQ2_XXS / UD-IQ1_M load with every expert tensor in one i-quant |
| `core/net_experts.cpp` `H = cpu::H` | the wire row is 2560 floats | `n_embd` is a run-time number (`NetExperts::open(..., n_embd)`, `ExpertServeCfg::n_embd`); 0 keeps 2560, so Flash-Next is byte-for-byte unchanged |
| `core/net_experts.cpp` `serve_experts(ExpertDispatch&)` | the server loop could only call Flash-Next's pack dispatch | `serve_experts_fn(cfg, compute)`: the loop (handshake, token, request validation) takes a callback; `net_experts_dispatch.cpp` is Flash-Next's callback, `native_rows.cpp` the qwen35moe one |
| `q36_engine.cu` MTP block | loaded whenever the file has one | dropped with a note when its experts have no GPU kernel (Q2_K in the Max files) |

New: `--remote-experts HOST:PORT=LO-HI[,...]` on `strata-q36` and on `strata` for a qwen35moe file (same flag, same SXP1 protocol as
Flash-Next, docs/REMOTE_EXPERTS.md); `strata-expert-server --native SHARD1.gguf --listen H:P --serve-layers LO-HI` serves a qwen35moe model
with no `--pack` (it repacks the served blocks' experts from the GGUF into RAM blobs [gate|up|down], the engine's own native blob, and
answers with the engine's CPU kernels and job grouping, so a row from the server is the row the conductor's CPU would compute). Blocks
`0..91` are the trunk, block `92` the MTP block. A block listed in `--remote-experts` keeps **no host copy** on the conductor (that is
the point for a 2.4T model), only its `--expert-slots` hottest-by-id experts in VRAM; the server computes the rest, in windows of at
most 8 tokens, with the window-invariant arithmetic (a row does not depend on its window). Block ranges follow `docs/REMOTE_EXPERTS.md`; a
server holds every expert of its blocks, and several hosts may share a block only if their `/E0-E1` expert ranges together cover it (the
conductor checks that at start: it has no copy to compute a missing row from).

## 3. The synthetic proof (no 500 GB)

`tools/make_synthetic_max.py OUT.gguf` writes a Max-shaped GGUF: the real tensor names, metadata keys and quantisation types (IQ1_M or
IQ2_XXS trunk experts, Q2_K MTP experts, Q5_K/Q6_K/Q8_0/Q4_K dense, F32 router/norms), 512 experts, top-10, head width 256, GDN
state 128, with random weights whose scale fields are set so activations stay finite through all blocks. Defaults: all 92 + 1 blocks,
hidden 256 (file 2.2 GiB). With flags it writes the **real widths** too (`--n-embd 8192 --n-ff 2048 --heads 64 --kv-heads 4
--ssm-k-heads 16 --ssm-v-heads 128`, two blocks, 11.5 GiB). `tests/qwen_max/run_synthetic.sh BUILD_CUDA BUILD_CPU [WORKDIR]` runs
both shapes on one 3090 (peak 3.6 GB of VRAM of ours on the real-width one) and compares 64 greedy tokens.

Measured 2026-11-03, nobara-pc (RTX 3090 #1, Ryzen 9 5950X, `strata-expert-server` on the same host, 8 workers, loopback), reference =
single process with the experts on the CPU pool (`STRATA_Q36_ARITH=wi`, `STRATA_Q36_FIXED_SLOTS=1`, `--expert-slots` 8 / 4), split = every
block's experts on the server:

| shape | experts | single process | split (loopback) | 64 generated tokens |
| --- | --- | --- | --- | --- |
| depth: 92 + 1 blocks, hidden 256, FF 256 | IQ1_M, 512 x top-10 | 118.6 tok/s | 77.5 tok/s | **identical** |
| wide: 2 blocks, hidden 8192, FF 2048, 64/4 heads, 16/128 GDN heads | IQ1_M, 512 x top-10 | 141.6 tok/s | 112.5 tok/s | **identical** |

(The split is slower on loopback because the conductor waits for a round trip it could have spent computing; the numbers prove the
path, not a speed.) CPU-only, in CI: `ctest -R qmax_native_rows` (needs numpy) runs `native_rows_test` on a 4 + 1 block synthetic file:
a row does not depend on its window; every row from a loopback expert server equals the in-process row byte for byte, for two trunk
blocks and the Q2_K MTP block; a conductor with another model's fingerprint is refused. The whole CPU-only suite passes as CI runs it (94 tests
without `isa_guard_check`, plus this one; 5 skipped for want of AVX-512 on the 5950X, as before), and `tools/net_experts_probe.py` (token, bad requests, busy, idle
connection) passes against the refactored Flash-Next server (`strata-expert-server --pack ...` on the Flash-Next pack).

Not shown by it: GPU and CPU arithmetic differ (as for every quantisation), so a hybrid run's text is not the all-in-VRAM text; on the
8-block file the logits of the two differ by at most 0.29 (largest logit 16.4, mean difference 0.043) with the same greedy token at all four
positions. The new GPU IQ1_M / IQ2_XXS down kernels are checked only that way (finite, same argmax), not against the real model.

## 4. Placement

`BigBrain/tools/xl/specs/qwen3.8-max_*.json` (DBN-AIO/AIO, `xl.model_spec/1`: 93 layers, 512 experts, top-10, the exact bytes above, rounded
up per expert; the MTP block's Q2_K experts are averaged into the 93 layers so the total is exact; wire row 32768 B = 8192 fp32) and
`python -m tools.xl.max_table` there. Conductor = the gaming PC (2 x 3090 over NVLink: the 35.7 GB of shared weights of all 93 blocks
must sit in VRAM, 43.2 GB usable pooled), reserves 10% VRAM / 20% RAM, experts never on NVMe, uniform popularity. Today's fleet = the five
hosts of the AIO test fixtures (gaming PC 64 GB RAM, rig 32 GB, stream PC 64 GB, T340 64 GB, portable); the hypothetical server is
CPU-only. Decimal GB.

| quant | experts | today's 5 hosts | + 768 GB RAM server | + 1 TB RAM server | RAM the new server needs |
| --- | --- | --- | --- | --- | --- |
| UD-Q1_0 | 360.4 GB | refused, short by **105.8 GB** | fits (server holds 106 GB) | fits (106 GB) | 132 GB |
| UD-IQ1_M | 527.1 GB | refused, short by **272.5 GB** | fits (server holds 273 GB) | fits (273 GB) | 341 GB |
| UD-IQ2_XXS | 619.7 GB | refused, short by **365.1 GB** | fits (server holds 365 GB) | fits (365 GB) | 457 GB |

("RAM needed" = shortfall / 0.8 usable.) On the fleet the other hosts hold: gaming PC 51 GB RAM + 11 GB VRAM copies, rig 26 GB RAM + 70 GB
VRAM copies, stream PC 51 GB + 16 GB VRAM, T340 51 GB, portable 6 GB VRAM; the new server takes whatever is left. KV cache and the
conductor's buffers are not in the spec. These sizes are the files' bytes; they say nothing about speed (the Ryzen-class CPU measured
2.9 tok/s for ten Flash-Next blocks on the rig's slower CPU; a 768 GB server's CPU decides Max's speed).

## 5. Blockers and what is left

1. **UD-Q1_0 does not load.** Its experts are ggml type 66, Unsloth's sub-IQ1_S type; the pinned llama.cpp (`third_party/ggml/VERSION.txt`,
   3cf03257) stops at type 42. `check_gguf_layout` refuses it cleanly ("has no known size"). The bytes are known exactly (placement above)
   but the block layout is not in the headers: it needs the type's definition from the upstream/Unsloth llama.cpp, then a CPU dot product
   (for the server) and optionally a GPU kernel. It is the smallest Max (397 GB), so it is the one worth doing.
2. **No MTP drafts on Max.** The MTP block's Q2_K experts have no GPU kernel; the block is left out with a note. The CPU server computes Q2_K
   (tested, block 4 of the synthetic). Serving block 92 remotely and keeping the MTP layer's attention on the GPU is the follow-up.
3. **Remote blocks have no fallback.** If the expert server dies the conductor exits with the error (Flash-Next recomputes locally; a
   2.4T conductor has no copy to recompute from). The net layer reconnects with backoff but the engine does not retry the layer yet.
4. **The prompt path** for a remote block goes through the same windows of 8 tokens as decode (pipelined 4 deep): correct, not fast. No
   MMQ streaming for remote blocks.
5. **Real weights, speed, quality**: not run. A first real run needs one of the larger files on a host with the RAM (the sizes above) and
   a one-time repack of the served blocks into RAM at server start.
6. `strata-expert-server` repacks each served block into anonymous RAM (mlock best effort, huge pages advised); a 768 GB host should raise
   `ulimit -l` or accept pageable memory.
