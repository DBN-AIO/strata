# Start strata-expert-server on the stream PC OUTSIDE the Claude desktop app's job object (docs/REMOTE_EXPERTS.md, "Windows").
# A process started directly from an agent shell dies with that app.  Pattern = TabbyAPI's switch_local.py: register a one-shot
# Task Scheduler task whose hidden pythonw launcher starts the server detached, run it, remove the task.
#   powershell -File tools\stream_pc\start_expert_server.ps1 -Layers 38-43 -Listen 0.0.0.0:7840
# Stop it with its PID only (never by image name):  Stop-Process -Id (Get-Content <Log>.pid)
param(
  [string]$Exe    = "C:\AIO-strata\build-cpu\strata-expert-server.exe",
  [string]$Pack   = "C:\AIO-strata-model\pack_q2",
  [string]$Native = "C:\AIO-strata-model\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
  [string]$Layers = "38-43",
  [string]$Listen = "0.0.0.0:7840",
  [string]$Log    = "C:\AIO-strata-model\expert-server.log",
  [string]$TokenFile = "C:\AIO-strata-model\expert.token",
  [int]$PoolWorkers = 0
)
$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "launch_expert_server.py"
$pyw = "C:\Python314\pythonw.exe"
$srvArgs = "--pack `"$Pack`" --native `"$Native`" --expert-server --listen $Listen --serve-layers $Layers"
if ($PoolWorkers -gt 0) { $srvArgs += " --pool-workers $PoolWorkers" }
$name = "AIO strata expert server one-shot start"
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$launcher`" `"$Exe`" `"$Log`" `"$TokenFile`" -- $srvArgs"
Register-ScheduledTask -TaskPath "\AIO\" -TaskName $name -Action $action -Force | Out-Null
Start-ScheduledTask -TaskPath "\AIO\" -TaskName $name
Start-Sleep -Seconds 3
Unregister-ScheduledTask -TaskPath "\AIO\" -TaskName $name -Confirm:$false
if (Test-Path "$Log.pid") { "started, pid " + (Get-Content "$Log.pid") + ", log $Log" } else { "no pid file: check the task" }
