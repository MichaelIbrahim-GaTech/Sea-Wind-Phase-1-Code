$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "run_full_end_to_end_from_inputs.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $script @args
exit $LASTEXITCODE
