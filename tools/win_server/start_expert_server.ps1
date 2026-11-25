# Start strata-expert-server detached, outside the calling shell's job object (docs/REMOTE_EXPERTS.md, "Windows").
# A process started directly from an agent shell dies with that app.  Register a one-shot Task Scheduler task whose hidden
# pythonw launcher starts the server detached, run it, remove the task.  Paths come from tools/win_server/local.json.
#   powershell -File tools\win_server\start_expert_server.ps1 -Layers 38-43
# The server refuses a non-loopback listen address without a token: put the token in the file named by "token_file".
# Stop it with its PID only (never by image name):  Stop-Process -Id (Get-Content <Log>.pid)
param(
  [string]$Layers = "38-43",
  [string]$Listen = "",       # default: local.json "listen" (this machine's LAN address; never all interfaces by default)
  [int]$PoolWorkers = 0
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$cfg = Read-LocalConfig
$Exe = Get-Setting $cfg "exe_cpu" ""
$Pack = Get-Setting $cfg "pack_qwen" ""
$Native = Get-Setting $cfg "native_qwen" ""
$Log = Get-Setting $cfg "log_qwen" ""
$TokenFile = Get-Setting $cfg "token_file" ""
$Listen = Get-Setting $cfg "listen" $Listen
$launcher = Join-Path $PSScriptRoot "launch_expert_server.py"
$pyw = Get-Setting $cfg "python_w" ""
$srvArgs = "--pack `"$Pack`" --native `"$Native`" --expert-server --listen $Listen --serve-layers $Layers"
if ($PoolWorkers -gt 0) { $srvArgs += " --pool-workers $PoolWorkers" }
$name = "strata expert server one-shot start"
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$launcher`" `"$Exe`" `"$Log`" `"$TokenFile`" -- $srvArgs"
Register-ScheduledTask -TaskPath "\strata\" -TaskName $name -Action $action -Force | Out-Null
Start-ScheduledTask -TaskPath "\strata\" -TaskName $name
Start-Sleep -Seconds 3
Unregister-ScheduledTask -TaskPath "\strata\" -TaskName $name -Confirm:$false
if (Test-Path "$Log.pid") { "started, pid " + (Get-Content "$Log.pid") + ", log $Log" } else { "no pid file: check the task" }
