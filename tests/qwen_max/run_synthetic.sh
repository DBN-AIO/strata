#!/bin/sh
# tests/qwen_max/run_synthetic.sh - the Qwen3.8-Max-shaped end-to-end proof on one machine with one NVIDIA card (docs/QWEN_MAX.md, "The
# synthetic proof").  It builds a Max-shaped synthetic GGUF, runs strata-q36 on it single-process, and again with the experts served
# by strata-expert-server over loopback, and compares the 64 generated tokens.
#
#   tests/qwen_max/run_synthetic.sh BUILD_CUDA_DIR BUILD_CPU_DIR [WORKDIR]
#     BUILD_CUDA_DIR  a CUDA configure with the strata-q36 target built
#     BUILD_CPU_DIR   a -DSTRATA_ENABLE_CUDA=OFF configure with strata-expert-server built
#     WORKDIR         where the model files go (default /tmp/qmax; the full-depth file is 2.2 GiB, the real-width one 11.5 GiB)
#   env: SHAPES="depth wide" (default: both), CUDA_VISIBLE_DEVICES (one card), PY (python with numpy), PORT (default 7861)
#
# "depth": 92 trunk blocks + the MTP block, 512 experts, top-10, n_embd 256 (every block of the real structure, small matrices).
# "wide":  2 blocks with the REAL Max widths (n_embd 8192, n_ff 2048, 64/4 heads, 16/128 GDN heads, 512 experts, top-10); small vocabulary.
# Both compare: reference = single process with the experts on the CPU pool (window-invariant arithmetic, fixed GPU slots), split = the
# same with every block's experts on the server.  Exit 0 only when the generated tokens are identical.
set -eu
CUDA=${1:?BUILD_CUDA_DIR}
CPU=${2:?BUILD_CPU_DIR}
WORK=${3:-/tmp/qmax}
PY=${PY:-python3}
PORT=${PORT:-7861}
SHAPES=${SHAPES:-"depth wide"}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$HERE")")
mkdir -p "$WORK"
rc=0
for shape in $SHAPES; do
  case $shape in
    depth) MODEL=$WORK/syn_depth.gguf; GEN="--layers 92"; LAST=91; PROMPT=248045,846,198,7734,264,2716,3255,883; SLOTS=8; CTX=2048;;
    wide)  MODEL=$WORK/syn_wide.gguf; GEN="--layers 2 --interval 2 --mtp 0 --n-embd 8192 --n-ff 2048 --heads 64 --kv-heads 4 --ssm-k-heads 16 --ssm-v-heads 128 --vocab 32768"; LAST=1; PROMPT=1000,2000,3000,4000,5000,6000,7000,8000; SLOTS=4; CTX=1024;;
    *) echo "unknown shape $shape"; exit 2;;
  esac
  [ -f "$MODEL" ] || $PY "$ROOT/tools/make_synthetic_max.py" "$MODEL" $GEN
  LOG=$WORK/server_$shape.log
  "$CPU/strata-expert-server" --native "$MODEL" --listen 127.0.0.1:$PORT --serve-layers 0-$LAST --pool-workers 8 > "$LOG" 2>&1 &
  SRV=$!
  i=0; until grep -q listening "$LOG" 2>/dev/null; do i=$((i+1)); [ $i -gt 300 ] && { echo "server did not start"; kill $SRV; exit 1; }; sleep 1; done
  COMMON="--model $MODEL --tokens $PROMPT --n 64 --max-context $CTX --no-warmup --expert-slots $SLOTS"
  REF=$(STRATA_Q36_FIXED_SLOTS=1 STRATA_Q36_ARITH=wi "$CUDA/strata-q36" $COMMON 2>&1 | grep '^gen:' || true)
  GOT=$(STRATA_Q36_FIXED_SLOTS=1 "$CUDA/strata-q36" $COMMON --remote-experts 127.0.0.1:$PORT=0-$LAST 2>&1 | grep '^gen:' || true)
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
  if [ -n "$REF" ] && [ "$REF" = "$GOT" ]; then echo "$shape: IDENTICAL ($(echo "$REF" | wc -w) tokens)"; else echo "$shape: DIFFERENT"; echo "ref:   $REF"; echo "split: $GOT"; rc=1; fi
done
exit $rc
