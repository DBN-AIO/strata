# Shared by start_expert_server.ps1 and start_dsv41_server.ps1: reads this machine's paths from tools/win_server/local.json.
# local.json is NOT tracked (see .gitignore); copy local.example.json to local.json and fill it in.
function Read-LocalConfig {
  $cfgPath = Join-Path $PSScriptRoot "local.json"
  if (-not (Test-Path $cfgPath)) { throw "tools/win_server/local.json is missing: copy local.example.json to local.json and fill in this machine's paths" }
  return Get-Content $cfgPath -Raw | ConvertFrom-Json
}
function Get-Setting($cfg, [string]$name, [string]$given) {
  if ($given) { return $given }
  $v = $cfg.$name
  if (-not $v) { throw "local.json has no '$name' (and no parameter was given)" }
  return [string]$v
}
