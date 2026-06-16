param(
    [switch]$RegenerateModelStages,
    [switch]$DoNotUseLockedStationRefine,
    [switch]$AlsoBuildNextCandidate
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:SEA_WINDS_WORKDIR = $workDir
if (-not $env:SEA_WINDS_OFFICIAL_PHASE1_DIR) {
    $env:SEA_WINDS_OFFICIAL_PHASE1_DIR = Join-Path $projectRoot "external\Hackathon-Sea-Winds-Predictions\phase_1"
}
$env:SEA_WINDS_KEEP_ZIP = "1"
$env:SEA_WINDS_LOW_RAM = "1"
$env:SEA_WINDS_FINALIZE_EXISTING = "0"

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

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required stage file is missing: $Path. Re-run with -RegenerateModelStages to rebuild model stages from official data/model caches."
    }
}

function Compact-CurrentPredictions {
    param(
        [string]$OutputCsv,
        [string]$OutputZip
    )
    Invoke-Step "Compact current predictions.csv -> $(Split-Path -Leaf $OutputCsv)" {
        & $venvPython (Join-Path $projectRoot "compact_predictions_zip.py") `
            --input (Join-Path $workDir "predictions.csv") `
            --output-csv $OutputCsv `
            --output-zip $OutputZip `
            --speed-dp 2 `
            --dir-dp 1
    }
}

function Restore-LockedStationRefine {
    $lockedCsv = Join-Path $workDir "baseline_station_refine_hybrid\predictions_station_refine_hybrid_bestknown.csv"
    $targetCsv = Join-Path $workDir "predictions_station_refine_hybrid_bestknown.csv"
    if (-not (Test-Path -LiteralPath $lockedCsv)) {
        Write-Host "Locked station-refine component not found; using current station-refine file."
        return
    }

    if (Test-Path -LiteralPath $targetCsv) {
        $lockedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $lockedCsv).Hash
        $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetCsv).Hash
        if ($lockedHash -eq $targetHash) {
            Write-Host "Locked station-refine component already active."
            return
        }

        $backupCsv = Join-Path $workDir "predictions_station_refine_hybrid_bestknown_regenerated_latest.csv"
        Copy-Item -LiteralPath $targetCsv -Destination $backupCsv -Force
        Write-Host "Backed up differing station-refine component to: $backupCsv"
    }

    Copy-Item -LiteralPath $lockedCsv -Destination $targetCsv -Force
    Write-Host "Restored locked station-refine component: $lockedCsv"
}

if ($RegenerateModelStages) {
    Invoke-Step "Generate station-refine base model stage" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot "run_v6_station_refine.ps1")
    }
    Compact-CurrentPredictions `
        -OutputCsv (Join-Path $workDir "predictions_station_refine_hybrid_bestknown.csv") `
        -OutputZip (Join-Path $workDir "submission_station_refine_hybrid_bestknown_compact.zip")

    Invoke-Step "Generate all-level direction model stage" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot "run_v6_direction_all_levels.ps1")
    }
    Compact-CurrentPredictions `
        -OutputCsv (Join-Path $workDir "predictions_direction_all_station_refine_compact.csv") `
        -OutputZip (Join-Path $workDir "submission_direction_all_station_refine_compact.zip")

    Invoke-Step "Generate proper full-refit model stage" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot "run_competitive_full_solution.ps1")
    }
}

if (-not $DoNotUseLockedStationRefine) {
    Invoke-Step "Restore locked station-refine component" {
        Restore-LockedStationRefine
    }
}

$stationRefineCsv = Join-Path $workDir "predictions_station_refine_hybrid_bestknown.csv"
$directionAllCsv = Join-Path $workDir "predictions_direction_all_station_refine_compact.csv"
$fullRefitCsv = Join-Path $workDir "predictions_proper_full_refit_v1_nsdirpost_compact.csv"
Require-File $stationRefineCsv
Require-File $directionAllCsv
Require-File $fullRefitCsv

Invoke-Step "Rebuild pressurefix station-long calibrated chain" {
    & $venvPython (Join-Path $projectRoot "build_pressurefix_station_long_calibrated.py")
}

Invoke-Step "Rebuild broad grid direction baseline" {
    & $venvPython (Join-Path $projectRoot "build_broad_grid_dir_candidate.py")
}

Invoke-Step "Assemble public-positive full-refit hybrid" {
    & $venvPython (Join-Path $projectRoot "build_public_positive_fullrefit_hybrid_candidate.py")
}

if ($AlsoBuildNextCandidate) {
    Invoke-Step "Build next NS pressure d7 surgical candidate" {
        & $venvPython (Join-Path $projectRoot "build_next_ns_pressure_d7_confirmed_candidate.py")
    }
}

$finalZip = Join-Path $workDir "submission_public_positive_fullrefit_hybrid_compact.zip"
Require-File $finalZip

Write-Host ""
Write-Host "Public-positive end-to-end submission:"
Write-Host $finalZip
if ($AlsoBuildNextCandidate) {
    Write-Host ""
    Write-Host "Next surgical candidate:"
    Write-Host (Join-Path $workDir "submission_next_ns_pressure_d7_confirmed_compact.zip")
}
