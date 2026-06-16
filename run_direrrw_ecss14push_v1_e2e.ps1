$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python build_direrrw_ecss14push_v1_candidate.py
if ($LASTEXITCODE -ne 0) { throw "build_direrrw_ecss14push_v1_candidate.py failed with exit code $LASTEXITCODE" }

python audit_final_submission.py `
  --zip runs/v6_pressure_speed/sub_direrrw_ecss14push_v1.zip `
  --baseline-csv runs/v6_pressure_speed/pred_dir_error_width_newsignal_v1.csv `
  --manifest runs/v6_pressure_speed/manifest_direrrw_ecss14push_v1_audit.json `
  --mode direrrw_ecss14push_v1
if ($LASTEXITCODE -ne 0) { throw "audit_final_submission.py failed with exit code $LASTEXITCODE" }
