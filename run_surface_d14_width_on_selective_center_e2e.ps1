param(
    [switch]$RegenerateBase,
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

Write-Host "[e2e] Surface d14 width on NS selective center candidate"
Write-Host "[e2e] RegenerateBase=$RegenerateBase RegenerateSelectiveCenter=$RegenerateSelectiveCenter GridPerAnchor=$GridPerAnchor"

$SelectiveCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_sfc14_selective_center.csv"
if ($RegenerateSelectiveCenter -or $RegenerateBase -or -not (Test-Path $SelectiveCsv)) {
    Write-Host "[e2e] regenerating NS surface d14 selective-center base first"
    $selScript = Join-Path $Root "run_ns_surface_d14_selective_center_e2e.ps1"
    $selArgs = @()
    if ($RegenerateBase) { $selArgs += "-RegenerateBase" }
    if ($RegenerateStationGateBase) { $selArgs += "-RegenerateStationGateBase" }
    if ($RegenerateStage2Base) { $selArgs += "-RegenerateStage2Base" }
    if ($RegenerateAnalogBase) { $selArgs += "-RegenerateAnalogBase" }
    if ($RegenerateRankAwareBase) { $selArgs += "-RegenerateRankAwareBase" }
    if ($RegenerateVectorAnenSource) { $selArgs += "-RegenerateVectorAnenSource" }
    if ($RegenerateModelStages) { $selArgs += "-RegenerateModelStages" }
    if ($StrictNoLockedStationRefine) { $selArgs += "-StrictNoLockedStationRefine" }
    if ($SkipBacktest) { $selArgs += "-SkipBacktest" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $selScript @selArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Selective-center generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $SelectiveCsv)) {
    throw "Selective-center CSV was not generated: $SelectiveCsv"
}

$OutCsv = Join-Path $Root "runs\v6_pressure_speed\pred_sfc14_selw.csv"
$OutZip = Join-Path $Root "runs\v6_pressure_speed\sub_sfc14_selw.zip"
$Summary = Join-Path $Root "runs\v6_pressure_speed\cv_sfc14_selw.csv"
$Manifest = Join-Path $Root "runs\v6_pressure_speed\manifest_sfc14_selw.json"

Write-Host "[e2e] applying surface d14 width gate on selective-center base"
& $Python (Join-Path $Root "build_surface_d14_direction_width_candidate.py") `
    --base-csv $SelectiveCsv `
    --output-csv $OutCsv `
    --output-zip $OutZip `
    --summary-csv $Summary `
    --manifest $Manifest `
    --grid-per-anchor $GridPerAnchor
if ($LASTEXITCODE -ne 0) {
    throw "Surface d14 width generation failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $OutZip)) {
    throw "Expected submission zip was not generated: $OutZip"
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
