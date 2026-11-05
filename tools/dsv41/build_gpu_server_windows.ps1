# The DeepSeek-V4.1 GPU expert server (strata-dsv41-expert-server-gpu.exe + ds4_gpu_rows_test.exe) for the stream PC's RTX 5070 Ti (sm_120)
# (BBXL, docs/DEEPSEEK41_SPLIT.md "Blackwell (sm_120) builds").
#
#   powershell -File tools\dsv41\build_gpu_server_windows.ps1 [-Src C:\AIO-strata] [-Build C:\AIO-strata\build-cuda-gpu] [-Check]
#
# -Check runs every prerequisite test and prints the exact missing step, builds nothing.  Without it the same checks run first and the
# build starts only when all pass.  Run the build through the capped CPU lane (the stream PC is the streaming machine, 70% cap):
#   MSYS_NO_PATHCONV=1 C:\Python314\python.exe "D:/AIStuff/AIO/AIO/AI Projects/AIOrchestrator/tools/cpu_lane/win_job.py" --cwd C:\AIO-strata -- powershell -File tools\dsv41\build_gpu_server_windows.ps1
# Installs nothing, changes no GPU setting, starts no server (start_dsv41_server.ps1 does that, outside the agent job object).
param(
  [string]$Src = "C:\AIO-strata",
  [string]$Build = "C:\AIO-strata\build-cuda-gpu",
  [string]$Arch = "120-real",
  [string]$Ggml = "C:\AIO-strata\build-cpu\_deps\strata_llamacpp-src",   # llama.cpp checkout; empty / missing: CMake fetches the pinned commit
  [string]$Pack = "C:\AIO-strata-model\dsv41-pack",
  [int]$Jobs = 4,
  [int]$MinFreeCGb = 15,       # the stream C: floor: the build must start with this much free, and refuses to continue under it
  [double]$MaxVramGb = 5,      # most VRAM the experts may take (the doc's Placement table: 5 GB free beside TabbyAPI's 9B)
  [double]$HeadroomGb = 1,     # left for the stream (NVENC / game capture), TabbyAPI's growth and the CUDA context
  [switch]$Check
)
$ErrorActionPreference = "Stop"
$missing = New-Object System.Collections.ArrayList
function Need($step) { [void]$missing.Add($step); Write-Host "MISSING: $step" -ForegroundColor Red }

# 1. disk: C: >= floor (D: is failing, never used)
$c = Get-PSDrive -Name C
$freeGb = [math]::Round($c.Free / 1GB, 1)
if ($freeGb -lt $MinFreeCGb) { Need "C: has $freeGb GB free, the floor is $MinFreeCGb GB: free space on C: first (the build needs about 3 GB; Jack approves deletions via the cleanup list)" } else { Write-Host "C: free $freeGb GB (floor $MinFreeCGb)" }

# 2. the source tree and the helper
if (-not (Test-Path "$Src\CMakeLists.txt")) { Need "no strata checkout at $Src (-Src <dir>)" }
$archCheck = Join-Path $PSScriptRoot "cuda_arch_check.py"
if (-not (Test-Path $archCheck)) { Need "tools\dsv41\cuda_arch_check.py is missing beside this script: update the checkout" }

# 3. CUDA toolkit (TODO-1276, Jack-only): CUDA_PATH is set by its installer
$cudaRoot = $env:CUDA_PATH
$nvcc = $null
if (-not $cudaRoot) { Need "no CUDA_PATH: the CUDA Toolkit is not installed (TODO-1276: install CUDA Toolkit 13.0 or newer, Visual Studio integration off)" }
else {
  $nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
  if (-not (Test-Path $nvcc)) { Need "CUDA_PATH=$cudaRoot has no bin\nvcc.exe: reinstall the toolkit with the Compiler component (TODO-1276)"; $nvcc = $null }
}
$py = "C:\Python314\python.exe"
if (-not (Test-Path $py)) { Need "no $py (the version check runs it)" }
$cudaVer = $null
if ($nvcc -and (Test-Path $py) -and (Test-Path $archCheck)) {
  & $py -I $archCheck --arch $Arch --nvcc $nvcc
  if ($LASTEXITCODE -ne 0) { Need "toolkit $cudaRoot cannot build -Arch $Arch (message above): install CUDA 13.0 or newer (TODO-1276), 12.8 is the floor for sm_120" }
  $m = [regex]::Match((& $nvcc --version | Out-String), "release (\d+)\.(\d+)")
  if ($m.Success) { $cudaVer = [version]("{0}.{1}" -f $m.Groups[1].Value, $m.Groups[2].Value) }
}

# 4. MSVC: CUDA 13.0-13.2 accept Visual Studio 2022 only, 13.3+ also 2026 (this PC has the 2026 Build Tools, MSVC 14.51)
$vc2022 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$vc2026 = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$vcvars = $null
if ($env:STRATA_VCVARS) { $vcvars = $env:STRATA_VCVARS }
elseif ($cudaVer -and $cudaVer -ge [version]"13.3") { $vcvars = if (Test-Path $vc2026) { $vc2026 } else { $vc2022 } }
else { $vcvars = $vc2022 }
if (-not (Test-Path $vcvars)) {
  if ($cudaVer -and $cudaVer -lt [version]"13.3") { Need "no ${vcvars}: CUDA $cudaVer accepts Visual Studio 2022 only: install the VS 2022 Build Tools (C++ workload) beside the 2026 ones, or take CUDA 13.3+ (TODO-1276)" }
  else { Need "no ${vcvars}: install the Visual Studio Build Tools C++ workload, or set STRATA_VCVARS to a vcvars64.bat" }
}

# 5. cmake and ninja (from the Build Tools or on PATH)
foreach ($t in @("cmake", "ninja")) { if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { Write-Host "note: $t is not on this shell's PATH; vcvars64.bat adds the Build Tools' copy" } }

# 6. the card, by UUID, and the VRAM the experts may take
$vramGb = 0
$uuid = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Need "no nvidia-smi: the NVIDIA driver is missing" }
else {
  $row = (& nvidia-smi --query-gpu=uuid,name,memory.total,memory.used,compute_cap --format=csv,noheader,nounits | Where-Object { $_ -match "5070 Ti" } | Select-Object -First 1)
  if (-not $row) { Need "no RTX 5070 Ti among the NVIDIA cards: this script sizes --gpu-vram-gb for that card" }
  else {
    $f = $row -split ",\s*"
    $uuid = $f[0]; $totalMib = [double]$f[2]; $usedMib = [double]$f[3]
    $vFreeGb = ($totalMib - $usedMib) / 1024
    $vramGb = [math]::Floor([math]::Min($MaxVramGb, $vFreeGb - $HeadroomGb) * 2) / 2     # whole or half GB, never above the cap
    Write-Host ("card {0} {1}: {2:N1} GB total, {3:N1} GB used now (TabbyAPI's 9B + the desktop), {4:N1} GB free" -f $uuid, $f[1], ($totalMib / 1024), ($usedMib / 1024), $vFreeGb)
    if ($vramGb -lt 1) { Need ("only {0:N1} GB of VRAM free beside TabbyAPI (headroom {1} GB kept): the experts would not fit a layer; unload a TabbyAPI model first or raise nothing here" -f $vFreeGb, $HeadroomGb) }
  }
}

if ($missing.Count -gt 0) {
  Write-Host ""
  Write-Host "REFUSED: $($missing.Count) prerequisite(s) missing (list above). Nothing was built." -ForegroundColor Red
  exit 2
}
Write-Host "all prerequisites present: CUDA $cudaVer at $cudaRoot, $vcvars"
if ($Check) { Write-Host "-Check: nothing built."; exit 0 }

# 7. configure + build (only the two GPU server targets), below normal priority
$ggArg = ""
if ($Ggml -and (Test-Path "$Ggml\ggml\CMakeLists.txt")) { $ggArg = "`"-DSTRATA_GGML_DIR=$Ggml`"" } else { Write-Host "no llama.cpp at ${Ggml}: CMake will fetch the pinned commit (network)" }
$cudaFwd = $cudaRoot -replace "\\", "/"
$bat = Join-Path $env:TEMP "strata_gpu_server_build.bat"
@"
@echo off
call "$vcvars" >nul 2>&1
cmake -G Ninja -S "$Src" -B "$Build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=$Arch" "-DCMAKE_CUDA_COMPILER=$cudaFwd/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$cudaFwd" $ggArg
if errorlevel 1 exit /b 1
if exist "$Build\strata-dsv41-expert-server-gpu.exe" del /q "$Build\strata-dsv41-expert-server-gpu.exe"
start "" /belownormal /wait /b cmake --build "$Build" --target strata-dsv41-expert-server-gpu ds4_gpu_rows_test -j $Jobs
if errorlevel 1 exit /b 1
exit /b 0
"@ | Set-Content -Path $bat -Encoding ASCII
& cmd /c $bat
if ($LASTEXITCODE -ne 0) { Write-Host "build failed (exit $LASTEXITCODE); batch file: $bat" -ForegroundColor Red; exit 1 }
$exe = Join-Path $Build "strata-dsv41-expert-server-gpu.exe"
if (-not (Test-Path $exe)) { Write-Host "no $exe" -ForegroundColor Red; exit 1 }
$freeAfter = [math]::Round((Get-PSDrive -Name C).Free / 1GB, 1)
if ($freeAfter -lt $MinFreeCGb) { Write-Host "warning: C: now has $freeAfter GB free, under the $MinFreeCGb GB floor" -ForegroundColor Yellow }

# 8. the smoke commands (printed, never run: a card test and a server start belong in Jack's / the operator's hands, outside the agent shell)
$rows = Join-Path $Build "ds4_gpu_rows_test.exe"
Write-Host ""
Write-Host "built $exe"
Write-Host "smoke 1 (rows vs the CPU lane on a synthetic pack, one card, ~1 GB VRAM; pack from tools\dsv41\make_synthetic_pack.py):"
Write-Host "  set CUDA_VISIBLE_DEVICES=$uuid"
Write-Host "  $rows --pack C:\AIO-strata-model\synth-pack --bench 200"
Write-Host "smoke 2 (the server, --gpu-vram-gb $vramGb = what is free beside TabbyAPI's 9B minus ${HeadroomGb} GB; --gpu-cpu-rest serves what does not fit by CPU):"
Write-Host "  powershell -File tools\stream_pc\start_dsv41_server.ps1 -Exe $exe -Pack $Pack -Layers 34-39 -Listen 0.0.0.0:7840 -ExtraArgs `"--expert-server-gpu --gpu-vram-gb $vramGb --gpu-cpu-rest`""
Write-Host "  (the server's own pin: this PC has one NVIDIA card, so ordinal 0 is it; in a foreground shell use CUDA_VISIBLE_DEVICES=$uuid)"
