# Start the DeepSeek-V4.1-Flash expert server detached (docs/DEEPSEEK41_SPLIT.md; same pattern as start_expert_server.ps1:
# a one-shot Task Scheduler task runs a hidden pythonw launcher that starts the server detached, then the task is removed).
# Paths come from tools/win_server/local.json.
#   powershell -File tools\win_server\start_dsv41_server.ps1 -Layers 26-34
# Stop it with its PID only (never by image name):  Stop-Process -Id (Get-Content <Log>.pid)
param(
  [string]$Layers = "26-34",
  [string]$Listen = "",       # default: local.json "listen" (this machine's LAN address); a non-loopback bind needs a token
  [string]$Exe = "",          # default: local.json "exe_dsv41" (the CUDA build takes the same flags)
  [int]$Threads = 0,
  [string]$ExtraArgs = ""     # e.g. "--expert-server-gpu --gpu-vram-gb 4 --gpu-cpu-rest" for the CUDA build (tools/dsv41/build_gpu_server_windows.ps1)
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$cfg = Read-LocalConfig
$Exe = Get-Setting $cfg "exe_dsv41" $Exe
$Pack = Get-Setting $cfg "pack_dsv41" ""
$Log = Get-Setting $cfg "log_dsv41" ""
$TokenFile = Get-Setting $cfg "token_file" ""
$Listen = Get-Setting $cfg "listen" $Listen
$launcher = Join-Path $PSScriptRoot "launch_expert_server.py"
$pyw = Get-Setting $cfg "python_w" ""
$srvArgs = "--pack `"$Pack`" --expert-server --listen $Listen --serve-layers $Layers"
if ($Threads -gt 0) { $srvArgs += " --cpu-threads $Threads" }
if ($ExtraArgs) { $srvArgs += " $ExtraArgs" }
$name = "strata dsv41 expert server one-shot start"
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$launcher`" `"$Exe`" `"$Log`" `"$TokenFile`" -- $srvArgs"
Register-ScheduledTask -TaskPath "\strata\" -TaskName $name -Action $action -Force | Out-Null
Start-ScheduledTask -TaskPath "\strata\" -TaskName $name
Start-Sleep -Seconds 3
Unregister-ScheduledTask -TaskPath "\strata\" -TaskName $name -Confirm:$false
if (Test-Path "$Log.pid") { "started, pid " + (Get-Content "$Log.pid") + ", log $Log" } else { "no pid file: check the task" }
