$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:SEA_WINDS_WORKDIR = $workDir
if (-not $env:SEA_WINDS_OFFICIAL_PHASE1_DIR) {
    $env:SEA_WINDS_OFFICIAL_PHASE1_DIR = Join-Path $projectRoot "external\Hackathon-Sea-Winds-Predictions\phase_1"
}

# Rule-compliant sources only: official provided train/inference files.
$env:SEA_WINDS_KEEP_ZIP = "1"
$env:SEA_WINDS_LOW_RAM = "1"
$env:SEA_WINDS_N_JOBS = "4"
$env:SEA_WINDS_ENABLE_STATIONS = "0"
$env:SEA_WINDS_DISABLE_STATIONS = "1"
$env:SEA_WINDS_STATION_DIR_POSTPROCESS = "1"
$env:SEA_WINDS_FINALIZE_EXISTING = "0"

# Strict two-stage setup:
#   1. tune and calibrate on 2019-2020 -> 2021;
#   2. refit selected model centers on all 2019-2021 while keeping calibration fixed.
$env:SEA_WINDS_MODEL_PROFILE = "proper_full_refit_v1"
$env:SEA_WINDS_RANDOM_SEED = "31"
$env:SEA_WINDS_TRAIN_WITH_2021 = "0"
$env:SEA_WINDS_RETRAIN_FULL = "1"

# Direct all-level grid models, LightGBM-only. This is the serious full solution,
# not a single-block probe.
$env:SEA_WINDS_DIRECT_SPEED_LEVELS = "10m,100m,1000,925,850,700,500"
$env:SEA_WINDS_DIRECT_DIR_LEVELS = "10m,100m,1000,925,850,700,500"
$env:SEA_WINDS_CATBOOST_SPEED_LEVELS = "none"
$env:SEA_WINDS_GRID_MAX_TRAIN_SAMPLES = "320000"
$env:SEA_WINDS_GRID_FEATURE_SUBSAMPLE = "120000"
$env:SEA_WINDS_GRID_DIR_FEATURE_SUBSAMPLE = "120000"
$env:SEA_WINDS_GRID_DIR_TRAIN_SUBSAMPLE = "300000"
$env:SEA_WINDS_LGB_SPEED_ITERS = "1100"
$env:SEA_WINDS_LGB_DIR_ITERS = "440"
$env:SEA_WINDS_LGB_SPEED_LEAVES = "63"
$env:SEA_WINDS_LGB_DIR_LEAVES = "63"

$modelCache = Join-Path $workDir "model_cache"
$sourceProfile = "quality_lgb_dirall"
$targetProfile = "proper_full_refit_v1"
$cacheCore = "v6_speed_10m_100m_1000_925_850_700_500__dir_10m_100m_1000_925_850_700_500__cb_none__profile_"
foreach ($region in @("north_sea", "east_china_sea")) {
    foreach ($kind in @("grid_speed", "grid_dir")) {
        $src = Join-Path $modelCache "$($region)_$($kind)_$($cacheCore)$($sourceProfile).pkl"
        $dst = Join-Path $modelCache "$($region)_$($kind)_$($cacheCore)$($targetProfile).pkl"
        if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
            Copy-Item -LiteralPath $src -Destination $dst
        }
    }
}

& $venvPython (Join-Path $projectRoot "sea_winds_solution_ephemeral_v6_pressure_speed.py")

$rawCsv = Join-Path $workDir "predictions.csv"
$compactCsv = Join-Path $workDir "predictions_proper_full_refit_v1_compact.csv"
$compactZip = Join-Path $workDir "submission_proper_full_refit_v1_compact.zip"
$finalCsv = Join-Path $workDir "predictions_proper_full_refit_v1_nsdirpost_compact.csv"
$finalZip = Join-Path $workDir "submission_proper_full_refit_v1_nsdirpost_compact.zip"

& $venvPython (Join-Path $projectRoot "compact_predictions_zip.py") `
    --input $rawCsv `
    --output-csv $compactCsv `
    --output-zip $compactZip `
    --speed-dp 2 `
    --dir-dp 1

& $venvPython (Join-Path $projectRoot "build_ns_grid_direction_next_candidate.py") `
    --base $compactCsv `
    --output-csv $finalCsv `
    --output-zip $finalZip

& $venvPython (Join-Path $projectRoot "report_submission_delta.py") `
    --base $compactCsv `
    --candidate $finalCsv

Write-Host ""
Write-Host "Final competitive submission:"
Write-Host $finalZip
