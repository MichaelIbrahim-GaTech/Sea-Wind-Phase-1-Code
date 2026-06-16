param(
    [switch]$RegenerateVectorAnenSource,
    [switch]$RegenerateModelStages,
    [switch]$StrictNoLockedStationRefine,
    [switch]$SkipCurrentBestRebuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"
$vectorSource = Join-Path $workDir "predictions_vector_anen_full_v1_compact.csv"
$finalZip = Join-Path $workDir "submission_rankaware_nsdir_stationmos_v1_compact.zip"
$manifest = Join-Path $workDir "rankaware_nsdir_stationmos_v1_manifest.json"

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

if (-not $SkipCurrentBestRebuild) {
    $finalArgs = @()
    if ($RegenerateModelStages) {
        $finalArgs += "-RegenerateModelStages"
    }
    if ($StrictNoLockedStationRefine) {
        $finalArgs += "-StrictNoLockedStationRefine"
    }
    Invoke-Step "Rebuild audited current-best base" {
        $finalScript = Join-Path $projectRoot "run_final_submission_e2e.ps1"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $finalScript @finalArgs
    }
}

if ($RegenerateVectorAnenSource -or -not (Test-Path -LiteralPath $vectorSource)) {
    Invoke-Step "Generate station vector-AnEn component" {
        & $venvPython (Join-Path $projectRoot "station_vector_anen.py")
    }
    Invoke-Step "Assemble AnEn hybrid component" {
        & $venvPython (Join-Path $projectRoot "build_anen_hybrid_v1_candidate.py")
    }
    Invoke-Step "Generate vector-AnEn full source" {
        & $venvPython (Join-Path $projectRoot "ns_pressure_d14_vector_anen.py")
    }
} else {
    Write-Host ""
    Write-Host "=== Use existing vector-AnEn source ==="
    Write-Host $vectorSource
}

Invoke-Step "Build rank-aware NS direction + station MOS candidate" {
    & $venvPython (Join-Path $projectRoot "build_rankaware_nsdir_stationmos_v1_candidate.py")
}

Write-Host ""
Write-Host "Rank-aware candidate zip:"
Write-Host $finalZip
Write-Host ""
Write-Host "Rank-aware manifest:"
Write-Host $manifest
