param(
    [switch]$RegenerateBase,
    [switch]$RegenerateBacktest,
    [switch]$RegenerateSelectiveCenter,
    [switch]$RegenerateStationGateBase,
    [switch]$RegenerateStage2Base,
    [switch]$RegenerateAnalogBase,
    [switch]$RegenerateRankAwareBase,
    [switch]$RegenerateVectorAnenSource,
    [switch]$RegenerateModelStages,
    [switch]$StrictNoLockedStationRefine,
    [switch]$SkipBacktest,
    [int]$GridPerAnchor = 450
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "[e2e] Seasonal NS public-positive fixed-block candidate"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_sfc14_selw.csv"
if ($RegenerateBase -or $RegenerateSelectiveCenter -or -not (Test-Path $BaseCsv)) {
    Write-Host "[e2e] regenerating current best surface d14 selective-width base first"
    $baseScript = Join-Path $Root "run_surface_d14_width_on_selective_center_e2e.ps1"
    $baseArgs = @()
    if ($RegenerateBase) { $baseArgs += "-RegenerateBase" }
    if ($RegenerateSelectiveCenter) { $baseArgs += "-RegenerateSelectiveCenter" }
    if ($RegenerateStationGateBase) { $baseArgs += "-RegenerateStationGateBase" }
    if ($RegenerateStage2Base) { $baseArgs += "-RegenerateStage2Base" }
    if ($RegenerateAnalogBase) { $baseArgs += "-RegenerateAnalogBase" }
    if ($RegenerateRankAwareBase) { $baseArgs += "-RegenerateRankAwareBase" }
    if ($RegenerateVectorAnenSource) { $baseArgs += "-RegenerateVectorAnenSource" }
    if ($RegenerateModelStages) { $baseArgs += "-RegenerateModelStages" }
    if ($StrictNoLockedStationRefine) { $baseArgs += "-StrictNoLockedStationRefine" }
    if ($SkipBacktest) { $baseArgs += "-SkipBacktest" }
    $baseArgs += "-GridPerAnchor"
    $baseArgs += "$GridPerAnchor"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $baseScript @baseArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Current best base generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $BaseCsv)) {
    throw "Expected base CSV was not generated: $BaseCsv"
}

$BacktestCsv = Join-Path $Root "runs\v6_pressure_speed\seasonal_direction_backtest_on_sfc14_selw.csv"
if ($RegenerateBacktest -or -not (Test-Path $BacktestCsv)) {
    Write-Host "[e2e] regenerating seasonal direction backtest table"
    & $Python (Join-Path $Root "seasonal_direction_backtest.py") `
        --base-csv $BaseCsv `
        --out-csv $BacktestCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Seasonal direction backtest generation failed with exit code $LASTEXITCODE"
    }
}

& $Python (Join-Path $Root "build_seasonal_fixed_blocks_candidate.py") `
    --profile nspos_b75 `
    --reason "Keep only public-positive seasonal blocks from sub_seas_b75.zip: NS pressure d7, NS pressure d14, NS surface d14." `
    --base-csv $BaseCsv `
    --backtest-csv $BacktestCsv `
    --out-csv (Join-Path $Root "runs\v6_pressure_speed\pred_nspos_b75.csv") `
    --out-zip (Join-Path $Root "runs\v6_pressure_speed\sub_nspos_b75.zip") `
    --manifest (Join-Path $Root "runs\v6_pressure_speed\manifest_nspos_b75.json") `
    --block "north_sea,pressure,7,blend_seasonal_w21_0.75" `
    --block "north_sea,pressure,14,blend_seasonal_w21_0.75" `
    --block "north_sea,surface,14,blend_seasonal_w21_0.75"
if ($LASTEXITCODE -ne 0) {
    throw "Seasonal fixed-block generation failed with exit code $LASTEXITCODE"
}

$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_nspos_b75.zip"
if (-not (Test-Path $OutZip)) {
    Write-Host "[e2e] no zip was emitted; inspect manifest_nspos_b75.json for gate-failure reason"
    exit 0
}

if ((Split-Path -Leaf $OutZip).Length -ge 64) {
    throw "Zip filename is too long for Codabench: $(Split-Path -Leaf $OutZip)"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($OutZip)
try {
    $names = @($zip.Entries | ForEach-Object { $_.FullName })
    if ($names.Count -ne 1 -or $names[0] -ne "predictions.csv") {
        throw "Zip must contain exactly root predictions.csv; found: $($names -join ', ')"
    }
} finally {
    $zip.Dispose()
}

Write-Host "[e2e] generated $OutZip"
