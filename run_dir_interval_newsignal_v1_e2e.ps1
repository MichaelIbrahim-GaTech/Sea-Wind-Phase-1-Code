$ErrorActionPreference = "Stop"

Write-Host "[e2e] direction interval new-signal v1"

$required = @(
    "build_dir_interval_newsignal_v1_candidate.py",
    "build_regime_newsignal_v1_candidate.py",
    "direction_anchor_backtest.py",
    "sea_winds_end_to_end_final.py",
    "runs\v6_pressure_speed\pred_ns_sfc14_rg90.csv",
    "runs\v6_pressure_speed\regime_newsignal_v1_rows_s180.parquet",
    "runs\v6_pressure_speed\model_cache\north_sea_grid_dir_v6_speed_10m_100m_1000_925_850_700_500__dir_10m_100m_1000_925_850_700_500__cb_none__profile_quality_lgb_dirall.pkl",
    "runs\v6_pressure_speed\model_cache\east_china_sea_grid_dir_v6_speed_10m_100m_1000_925_850_700_500__dir_10m_100m_1000_925_850_700_500__cb_none__profile_quality_lgb_dirall.pkl"
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required input: $path"
    }
}

@'
import importlib.util
missing = [m for m in ("pandas", "numpy", "pyarrow") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing))
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "Dependency check failed with exit code $LASTEXITCODE"
}

python -m py_compile build_dir_interval_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Compile failed with exit code $LASTEXITCODE"
}

python build_dir_interval_newsignal_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "Builder failed with exit code $LASTEXITCODE"
}

$manifest = "runs\v6_pressure_speed\manifest_dir_interval_newsignal_v1.json"
if (-not (Test-Path $manifest)) {
    throw "Missing manifest: $manifest"
}

Write-Host "[e2e] complete: $manifest"
