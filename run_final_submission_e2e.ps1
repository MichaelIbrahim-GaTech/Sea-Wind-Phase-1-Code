param(
    [switch]$RegenerateModelStages,
    [switch]$StrictNoLockedStationRefine
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"
$finalZip = Join-Path $workDir "submission_station_lgbm_ecs_d1_dir_cv_compact.zip"
$baselineCsv = Join-Path $workDir "predictions_public_positive_fullrefit_hybrid_compact.csv"
$manifest = Join-Path $workDir "final_submission_manifest.json"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing Python environment: $venvPython"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    $global:LASTEXITCODE = 0
    & $Command
    if (-not $?) {
        throw "$Name failed."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-CommandStep {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments = @()
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    $global:LASTEXITCODE = 0
    & $Command @Arguments
    if (-not $?) {
        throw "$Name failed."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$baseArgs = @()
if ($RegenerateModelStages) {
    $baseArgs += "-RegenerateModelStages"
}
if ($StrictNoLockedStationRefine) {
    $baseArgs += "-DoNotUseLockedStationRefine"
}

Invoke-Step "Generate public-positive baseline stage" {
    $publicPositiveScript = Join-Path $projectRoot "run_public_positive_fullrefit_hybrid_e2e.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $publicPositiveScript @baseArgs
}

Invoke-Step "Apply CV-gated ECS station d1 direction model" {
    & $venvPython (Join-Path $projectRoot "build_station_lgbm_ecs_d1_direction_cv_candidate.py")
}

Invoke-Step "Audit final submission zip and write manifest" {
    $mode = if ($StrictNoLockedStationRefine) { "strict-regenerated-stage-e2e" } else { "locked-stage-e2e" }
    & $venvPython (Join-Path $projectRoot "audit_final_submission.py") `
        --zip $finalZip `
        --baseline-csv $baselineCsv `
        --manifest $manifest `
        --mode $mode
}

Write-Host ""
Write-Host "Final submission zip:"
Write-Host $finalZip
Write-Host ""
Write-Host "Final manifest:"
Write-Host $manifest
