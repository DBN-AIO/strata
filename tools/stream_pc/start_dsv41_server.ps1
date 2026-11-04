# Start the DeepSeek-V4.1-Flash expert server on the stream PC OUTSIDE the Claude desktop app's job object
# (docs/DEEPSEEK41_SPLIT.md; same pattern as start_expert_server.ps1: a one-shot Task Scheduler task runs a hidden pythonw launcher
# that starts the server detached, then the task is removed).
#   powershell -File tools\stream_pc\start_dsv41_server.ps1 -Layers 26-34 -Listen 0.0.0.0:7840
# Stop it with its PID only (never by image name):  Stop-Process -Id (Get-Content <Log>.pid)
# Stream mode stops it (the other hosts take its layers, or the conductor runs those layers itself).
param(
  [string]$Exe    = "C:\AIO-strata\build-cpu\strata-dsv41-expert-server.exe",   # the CUDA build is C:\AIO-strata\build-cuda\strata-dsv41.exe (same flags)
  [string]$Pack   = "C:\AIO-strata-model\dsv41-pack",
  [string]$Layers = "26-34",
  [string]$Listen = "0.0.0.0:7840",
  [string]$Log    = "C:\AIO-strata-model\dsv41-expert-server.log",
  [string]$TokenFile = "C:\AIO-strata-model\expert.token",
  [int]$Threads = 0,
  [string]$ExtraArgs = ""   # e.g. "--expert-server-gpu --gpu-vram-gb 4 --gpu-cpu-rest" for the CUDA build (tools\dsv41uild_gpu_server_windows.ps1)
)
$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "launch_expert_server.py"
$pyw = "C:\Python314\pythonw.exe"
$srvArgs = "--pack `"$Pack`" --expert-server --listen $Listen --serve-layers $Layers"
if ($Threads -gt 0) { $srvArgs += " --cpu-threads $Threads" }
if ($ExtraArgs) { $srvArgs += " $ExtraArgs" }
$name = "AIO strata dsv41 expert server one-shot start"
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$launcher`" `"$Exe`" `"$Log`" `"$TokenFile`" -- $srvArgs"
Register-ScheduledTask -TaskPath "\AIO\" -TaskName $name -Action $action -Force | Out-Null
Start-ScheduledTask -TaskPath "\AIO\" -TaskName $name
Start-Sleep -Seconds 3
Unregister-ScheduledTask -TaskPath "\AIO\" -TaskName $name -Confirm:$false
if (Test-Path "$Log.pid") { "started, pid " + (Get-Content "$Log.pid") + ", log $Log" } else { "no pid file: check the task" }
