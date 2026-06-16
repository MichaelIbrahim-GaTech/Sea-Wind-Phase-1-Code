$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_sfc14_rg90.csv"
if (-not (Test-Path $BaseCsv)) {
    throw "Missing current-best base CSV: $BaseCsv"
}

$RowCache = Join-Path $Root "runs\v6_pressure_speed\regime_newsignal_v1_rows_s60.parquet"
if (-not (Test-Path $RowCache)) {
    $env:SEA_WINDS_REGIME_NEW_SIGNAL_SAMPLE_PER_DATE = "60"
    & $Python (Join-Path $Root "build_regime_newsignal_v1_candidate.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create row cache, builder failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[e2e] learned residual new-signal v1"
& $Python -m py_compile (Join-Path $Root "build_learned_residual_newsignal_v1_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "Compile failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $Root "build_learned_residual_newsignal_v1_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "Builder failed with exit code $LASTEXITCODE"
}

$Manifest = Join-Path $Root "runs\v6_pressure_speed\manifest_learned_residual_newsignal_v1.json"
if (-not (Test-Path $Manifest)) {
    throw "Expected manifest was not written: $Manifest"
}

$manifestJson = Get-Content $Manifest -Raw | ConvertFrom-Json
if ($manifestJson.status -eq "submission_written") {
    Write-Host "[e2e] submission written: $($manifestJson.submission.zip)"
} else {
    Write-Host "[e2e] no submission emitted: $($manifestJson.reason)"
}
