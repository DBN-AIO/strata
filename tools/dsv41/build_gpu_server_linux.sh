#!/bin/bash
# The DeepSeek-V4.1 GPU expert server (strata-dsv41-expert-server-gpu + ds4_gpu_rows_test) for NVIDIA on Linux, sm_86 + sm_120 (BBXL,
# docs/DEEPSEEK41_SPLIT.md "Blackwell (sm_120) builds").  Built for the AI rig: 3090 / 3060 (sm_86) and 5070 (sm_120).
#
#   CUDA=<toolkit root> B=<build dir> [SRC=<checkout>] [ARCH="86-real;120-real"] [GG=<llama.cpp checkout>] [JOBS=3] \
#       tools/dsv41/build_gpu_server_linux.sh
#
# CUDA is required and is never guessed: a user-local toolkit (e.g. /mnt/models/cuda-13.0, see the doc) or /usr/local/cuda-13.0.  The rig's own
# /usr/bin/nvcc is Ubuntu's 12.4 (sm_86 only): with it ARCH=86-real is the only thing that builds, and ARCH with 120 is refused below.
# Refuses, naming the missing step, when: nvcc is missing or too old for ARCH, the driver is older than the toolkit, the build dir's
# filesystem has under MIN_FREE_GB (default 6) free, or no host compiler fits the toolkit (12.x: g++-13; 13.x: g++-15 or g++-14).
# Reads only; changes no GPU setting, installs nothing, never touches the driver.  Last line printed: the binary.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$(cd "$here/../.." && pwd)}
B=${B:?B=<build dir> (not on a full disk: the rig /home is ~99% full, /mnt/models has room)}
CUDA=${CUDA:?CUDA=<toolkit root, e.g. /mnt/models/cuda-13.0>: the rig has only Ubuntu nvcc 12.4, which cannot build sm_120 (doc: Blackwell builds)}
ARCH=${ARCH:-86-real;120-real}
JOBS=${JOBS:-3}
MIN_FREE_GB=${MIN_FREE_GB:-6}
CMAKE=${CMAKE:-$(command -v cmake || true)}
[ -n "$CMAKE" ] || CMAKE=$HOME/tools/cmake-3.31.6-linux-x86_64/bin/cmake
die() { echo "build_gpu_server_linux.sh: $*" >&2; exit 2; }

[ -x "$CUDA/bin/nvcc" ] || die "no $CUDA/bin/nvcc: install the CUDA toolkit there (runfile --toolkit --toolkitpath=$CUDA --silent, no driver; Jack-only step, doc \"Blackwell (sm_120) builds\")"
[ -x "$CMAKE" ] || die "no cmake (CMAKE=<path>; the T340 has ~/tools/cmake-3.31.6-linux-x86_64/bin/cmake)"
[ -f "$SRC/CMakeLists.txt" ] || die "SRC=$SRC is not a strata checkout"

# 1. the toolkit can build ARCH (same table the CMake configure uses)
python3 "$here/cuda_arch_check.py" --arch "$ARCH" --nvcc "$CUDA/bin/nvcc" || die "toolkit $CUDA cannot build ARCH=$ARCH (message above)"
ver=$("$CUDA/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -1)
major=${ver%%.*}

# 2. the driver is new enough for that toolkit (the toolkit's runtime must not be newer than the driver's CUDA)
if command -v nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\?Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -1)
  if [ -n "$drv" ]; then
    [ "$(printf '%s\n%s\n' "$ver" "$drv" | sort -V | tail -1)" = "$drv" ] || die "driver supports CUDA $drv only, toolkit is $ver: use a toolkit <= $drv (a driver update is Jack-only)"
    echo "driver CUDA $drv >= toolkit $ver"
  fi
fi

# 3. host compiler: nvcc 12.x takes gcc <= 13 (Ubuntu's gcc 15 is refused), 13.x takes up to 15
if [ -z "${HOSTCXX:-}" ]; then
  if [ "$major" -ge 13 ]; then cands="g++-15 g++-14 g++-13 g++"; else cands="g++-13 g++-12 g++-11"; fi
  for c in $cands; do command -v "$c" >/dev/null 2>&1 && { HOSTCXX=$c; break; }; done
fi
[ -n "${HOSTCXX:-}" ] || die "no host compiler for CUDA $ver (needs g++-13 for 12.x, g++-14/15 for 13.x; HOSTCXX=<g++>)"
HOSTCC=${HOSTCXX/g++/gcc}
echo "host compiler $HOSTCXX"

# 4. disk: the build dir's filesystem (explicit path: a bare df hangs on the rig's stale NFS)
mkdir -p "$B"
free_kb=$(df -Pk "$B" | awk 'NR==2{print $4}')
[ "$free_kb" -ge $((MIN_FREE_GB * 1024 * 1024)) ] || die "only $((free_kb / 1048576)) GB free under $B (need $MIN_FREE_GB): pick B on /mnt/models (explicit path), never the full /home"

# 5. configure and build (only the two GPU server targets; the engine itself is not needed)
command -v ninja >/dev/null 2>&1 || die "no ninja on PATH"
if [ -n "${GG:-}" ]; then ggarg="-DSTRATA_GGML_DIR=$GG"; else ggarg=""; fi
export PATH=$CUDA/bin:$PATH
rm -f "$B/CMakeCache.txt" "$B/strata-dsv41-expert-server-gpu" "$B/ds4_gpu_rows_test"
nice -n 19 "$CMAKE" -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" \
  "-DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc" "-DCUDAToolkit_ROOT=$CUDA" "-DCMAKE_C_COMPILER=$HOSTCC" "-DCMAKE_CXX_COMPILER=$HOSTCXX" \
  "-DCMAKE_CUDA_HOST_COMPILER=$HOSTCXX" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF '-DCMAKE_INSTALL_RPATH=$ORIGIN' $ggarg
nice -n 19 "$CMAKE" --build "$B" --target strata-dsv41-expert-server-gpu ds4_gpu_rows_test -j "$JOBS"
bin=$B/strata-dsv41-expert-server-gpu
[ -x "$bin" ] || die "no $bin"

# 6. the binary carries the SASS that was asked for
if command -v cuobjdump >/dev/null 2>&1 || [ -x "$CUDA/bin/cuobjdump" ]; then
  cuo=$CUDA/bin/cuobjdump; [ -x "$cuo" ] || cuo=cuobjdump
  have=$("$cuo" --list-elf "$bin" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
  echo "SASS in the binary: $have"
  for a in $(echo "$ARCH" | tr ';,' '  '); do
    n=${a%%-*}; case "$n" in [0-9]*) case " $have" in *" sm_$n "*) ;; *) die "binary has no sm_$n SASS (have: $have)";; esac;; esac
  done
fi

# 7. the cards, by UUID (never by index), and the smoke command for each architecture built
echo "cards (pin with CUDA_VISIBLE_DEVICES=<uuid>):"
nvidia-smi --query-gpu=uuid,name,compute_cap,memory.total,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
cat >&2 <<EOF
smoke (needs a card with free VRAM; the rig's TabbyAPI holds most of them, see the doc's Placement table; pick the 5070 by its UUID):
  python3 $SRC/tools/dsv41/make_synthetic_pack.py <pack dir>            # once, small
  CUDA_VISIBLE_DEVICES=<5070 uuid> $B/ds4_gpu_rows_test --pack <pack dir> --bench 200
  CUDA_VISIBLE_DEVICES=<5070 uuid> STRATA_EXPERT_TOKEN=... $bin --pack <pack dir> --listen 127.0.0.1:7841 \\
      --expert-server-gpu --serve-layers 0-1 --gpu-vram-gb 3 --gpu-cpu-rest
EOF
echo "$bin"
