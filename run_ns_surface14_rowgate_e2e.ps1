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

Write-Host "[e2e] NS surface d14 pruned row-gated seasonal direction candidate"

$OrigBaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_sfc14_selw.csv"
$ConfirmedBaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_nspos_b75.csv"
if ($RegenerateBase -or -not (Test-Path $OrigBaseCsv) -or -not (Test-Path $ConfirmedBaseCsv)) {
    Write-Host "[e2e] regenerating confirmed nspos base first"
    $baseScript = Join-Path $Root "run_seasonal_ns_positive_e2e.ps1"
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
        throw "nspos base generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $OrigBaseCsv)) {
    throw "Expected original base CSV was not generated: $OrigBaseCsv"
}
if (-not (Test-Path $ConfirmedBaseCsv)) {
    throw "Expected confirmed base CSV was not generated: $ConfirmedBaseCsv"
}

$OutCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_sfc14_rg.csv"
$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_ns_sfc14_rg.zip"
$Manifest = Join-Path $Root "runs\v6_pressure_speed\manifest_ns_sfc14_rg.json"

& $Python (Join-Path $Root "build_seasonal_rowgate_candidate.py") `
    --orig-base-csv $OrigBaseCsv `
    --confirmed-base-csv $ConfirmedBaseCsv `
    --out-csv $OutCsv `
    --out-zip $OutZip `
    --manifest $Manifest `
    --target "north_sea,surface,14"
if ($LASTEXITCODE -ne 0) {
    throw "NS surface d14 row-gated generation failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $OutZip)) {
    Write-Host "[e2e] no zip was emitted; inspect manifest_ns_sfc14_rg.json for gate-failure reason"
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

& $Python (Join-Path $Root "audit_final_submission.py") `
    --zip $OutZip `
    --baseline-csv $ConfirmedBaseCsv `
    --manifest (Join-Path $Root "runs\v6_pressure_speed\audit_ns_sfc14_rg.json") `
    --mode "ns-surface-d14-rowgate-pruned-e2e"
if ($LASTEXITCODE -ne 0) {
    throw "Audit failed with exit code $LASTEXITCODE"
}

Write-Host "[e2e] generated $OutZip"
