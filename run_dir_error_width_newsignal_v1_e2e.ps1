$ErrorActionPreference = "Stop"

Write-Host "[e2e] direction error-width new-signal v1"

$required = @(
    "build_dir_error_width_newsignal_v1_candidate.py",
    "build_dir_interval_newsignal_v1_candidate.py",
    "build_feature_rich_newsignal_v1_candidate.py",
    "runs\v6_pressure_speed\pred_ns_sfc14_rg90.csv",
    "runs\v6_pressure_speed\feature_rich_newsignal_v1_rows_s180.parquet",
    "runs\v6_pressure_speed\phase1_dataset\features\inference_window_1_north_sea.parquet",
    "runs\v6_pressure_speed\phase1_dataset\features\inference_window_1_east_china_sea.parquet"
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required input: $path"
    }
}

@'
import importlib.util
missing = [m for m in ("lightgbm", "pandas", "numpy", "pyarrow") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing))
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "Dependency check failed with exit code $LASTEXITCODE"
}

python -m py_compile build_dir_error_width_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Compile failed with exit code $LASTEXITCODE"
}

python build_dir_error_width_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Builder failed with exit code $LASTEXITCODE"
}

$manifest = "runs\v6_pressure_speed\manifest_dir_error_width_newsignal_v1.json"
if (-not (Test-Path $manifest)) {
    throw "Missing manifest: $manifest"
}

Write-Host "[e2e] complete: $manifest"
