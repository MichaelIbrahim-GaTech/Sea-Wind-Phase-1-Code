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

Write-Host "[e2e] ECS surface d14 shrink-80 candidate"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ecsp14_s70.csv"
if ($RegenerateBase -or -not (Test-Path $BaseCsv)) {
    Write-Host "[e2e] generating current confirmed base first"
    $baseScript = Join-Path $Root "run_ecsp14_s70_e2e.ps1"
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
        throw "Current base generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $BaseCsv)) {
    throw "Expected base CSV was not generated: $BaseCsv"
}

& $Python (Join-Path $Root "build_ecs_surface_d14_w80_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "ECS surface d14 shrink-80 build failed with exit code $LASTEXITCODE"
}

$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_ecss14w80.zip"
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
