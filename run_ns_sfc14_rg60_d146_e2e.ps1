param(
    [switch]$RegenerateBase,
    [switch]$RegenerateOrigBase,
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

Write-Host "[e2e] NS surface d14 row-gate delta60 on promoted d146+rg45 base"

$ConfirmedBaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_sfc14rg_d146.csv"
if ($RegenerateBase -or -not (Test-Path $ConfirmedBaseCsv)) {
    Write-Host "[e2e] generating promoted d146+rg45 base first"
    $baseScript = Join-Path $Root "run_sfc14rg_d146_e2e.ps1"
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
        throw "Promoted d146+rg45 base generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $ConfirmedBaseCsv)) {
    throw "Expected promoted base CSV was not generated: $ConfirmedBaseCsv"
}

$OrigBaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_sfc14_selw.csv"
if ($RegenerateOrigBase -or -not (Test-Path $OrigBaseCsv)) {
    Write-Host "[e2e] generating NS surface d14 selective-width source first"
    $origScript = Join-Path $Root "run_surface_d14_width_on_selective_center_e2e.ps1"
    $origArgs = @()
    if ($RegenerateOrigBase) { $origArgs += "-RegenerateBase" }
    if ($RegenerateSelectiveCenter) { $origArgs += "-RegenerateSelectiveCenter" }
    if ($RegenerateStationGateBase) { $origArgs += "-RegenerateStationGateBase" }
    if ($RegenerateStage2Base) { $origArgs += "-RegenerateStage2Base" }
    if ($RegenerateAnalogBase) { $origArgs += "-RegenerateAnalogBase" }
    if ($RegenerateRankAwareBase) { $origArgs += "-RegenerateRankAwareBase" }
    if ($RegenerateVectorAnenSource) { $origArgs += "-RegenerateVectorAnenSource" }
    if ($RegenerateModelStages) { $origArgs += "-RegenerateModelStages" }
    if ($StrictNoLockedStationRefine) { $origArgs += "-StrictNoLockedStationRefine" }
    if ($SkipBacktest) { $origArgs += "-SkipBacktest" }
    $origArgs += "-GridPerAnchor"
    $origArgs += "$GridPerAnchor"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $origScript @origArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NS surface d14 selective-width source generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $OrigBaseCsv)) {
    throw "Expected original base/source CSV was not generated: $OrigBaseCsv"
}

& $Python (Join-Path $Root "build_ns_sfc14_rg60_d146_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "NS surface d14 rg60 candidate build failed with exit code $LASTEXITCODE"
}

$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_nssfc14rg60.zip"
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
