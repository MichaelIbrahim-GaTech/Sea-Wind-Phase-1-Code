$ErrorActionPreference = "Stop"

Write-Host "[e2e] LightGBM current-base residual new-signal v1"

$required = @(
    "build_lgbm_base_residual_newsignal_v1_candidate.py",
    "build_learned_residual_newsignal_v1_candidate.py",
    "build_regime_newsignal_v1_candidate.py",
    "runs\v6_pressure_speed\pred_ns_sfc14_rg90.csv",
    "runs\v6_pressure_speed\phase1_dataset\features\train_north_sea.parquet",
    "runs\v6_pressure_speed\phase1_dataset\features\train_east_china_sea.parquet"
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required input: $path"
    }
}

@'
import importlib.util
if importlib.util.find_spec("lightgbm") is None:
    raise SystemExit("Missing dependency: lightgbm")
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "Dependency check failed with exit code $LASTEXITCODE"
}

python -m py_compile build_lgbm_base_residual_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Compile failed with exit code $LASTEXITCODE"
}

python build_lgbm_base_residual_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Builder failed with exit code $LASTEXITCODE"
}

$manifest = "runs\v6_pressure_speed\manifest_lgbm_base_residual_newsignal_v1.json"
if (-not (Test-Path $manifest)) {
    throw "Missing manifest: $manifest"
}

Write-Host "[e2e] complete: $manifest"
