$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:SEA_WINDS_WORKDIR = Join-Path $projectRoot "runs\v6_pressure_speed"
if (-not $env:SEA_WINDS_OFFICIAL_PHASE1_DIR) {
    $env:SEA_WINDS_OFFICIAL_PHASE1_DIR = Join-Path $projectRoot "external\Hackathon-Sea-Winds-Predictions\phase_1"
}

# Direction refinement: train direct direction models for all scored vertical
# levels while reusing the existing quality_lgb speed and station model caches.
$env:SEA_WINDS_LOW_RAM = "1"
$env:SEA_WINDS_N_JOBS = "4"
$env:SEA_WINDS_KEEP_ZIP = "1"
$env:SEA_WINDS_ENABLE_STATIONS = "1"
$env:SEA_WINDS_STATION_DIR_POSTPROCESS = "1"
$env:SEA_WINDS_MODEL_PROFILE = "quality_lgb_dirall"

$env:SEA_WINDS_DIRECT_SPEED_LEVELS = "10m,100m,1000,925,850,700,500"
$env:SEA_WINDS_DIRECT_DIR_LEVELS = "10m,100m,1000,925,850,700,500"
$env:SEA_WINDS_CATBOOST_SPEED_LEVELS = "none"

# Keep direction models moderately strong but still tractable locally.
$env:SEA_WINDS_GRID_DIR_FEATURE_SUBSAMPLE = "90000"
$env:SEA_WINDS_GRID_DIR_TRAIN_SUBSAMPLE = "220000"
$env:SEA_WINDS_LGB_DIR_ITERS = "350"

& (Join-Path $projectRoot ".venv\Scripts\python.exe") (Join-Path $projectRoot "sea_winds_solution_ephemeral_v6_pressure_speed.py")
