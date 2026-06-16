param(
    [switch]$RegenerateBase,
    [switch]$RegenerateRowgateSource,
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

Write-Host "[e2e] NS surface d14 row-gate layer on promoted d146 base"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ecs14d146.csv"
if ($RegenerateBase -or -not (Test-Path $BaseCsv)) {
    Write-Host "[e2e] generating promoted d146 base first"
    $baseScript = Join-Path $Root "run_ecs14d146_e2e.ps1"
    $baseArgs = @()
    if ($RegenerateBase) { $baseArgs += "-RegenerateBase" }
    if ($RegenerateBacktest) { $baseArgs += "-RegenerateBacktest" }
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
        throw "Promoted d146 base generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $BaseCsv)) {
    throw "Expected promoted d146 base CSV was not generated: $BaseCsv"
}

$PatchSourceCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_sfc14_rg.csv"
if ($RegenerateRowgateSource -or -not (Test-Path $PatchSourceCsv)) {
    Write-Host "[e2e] generating NS surface d14 row-gate source first"
    $sourceScript = Join-Path $Root "run_ns_surface14_rowgate_e2e.ps1"
    $sourceArgs = @()
    if ($RegenerateRowgateSource) { $sourceArgs += "-RegenerateBase" }
    if ($RegenerateBacktest) { $sourceArgs += "-RegenerateBacktest" }
    if ($RegenerateSelectiveCenter) { $sourceArgs += "-RegenerateSelectiveCenter" }
    if ($RegenerateStationGateBase) { $sourceArgs += "-RegenerateStationGateBase" }
    if ($RegenerateStage2Base) { $sourceArgs += "-RegenerateStage2Base" }
    if ($RegenerateAnalogBase) { $sourceArgs += "-RegenerateAnalogBase" }
    if ($RegenerateRankAwareBase) { $sourceArgs += "-RegenerateRankAwareBase" }
    if ($RegenerateVectorAnenSource) { $sourceArgs += "-RegenerateVectorAnenSource" }
    if ($RegenerateModelStages) { $sourceArgs += "-RegenerateModelStages" }
    if ($StrictNoLockedStationRefine) { $sourceArgs += "-StrictNoLockedStationRefine" }
    if ($SkipBacktest) { $sourceArgs += "-SkipBacktest" }
    $sourceArgs += "-GridPerAnchor"
    $sourceArgs += "$GridPerAnchor"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $sourceScript @sourceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NS surface d14 row-gate source generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $PatchSourceCsv)) {
    throw "Expected row-gate source CSV was not generated: $PatchSourceCsv"
}

& $Python (Join-Path $Root "build_sfc14rg_d146_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "NS surface d14 row-gate d146 layer build failed with exit code $LASTEXITCODE"
}

$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_sfc14rgd146.zip"
if (-not (Test-Path $OutZip)) {
    throw "Expected zip was not generated: $OutZip"
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
