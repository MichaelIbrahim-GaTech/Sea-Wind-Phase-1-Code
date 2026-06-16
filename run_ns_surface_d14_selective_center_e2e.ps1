param(
    [switch]$RegenerateBase,
    [switch]$RegenerateStationGateBase,
    [switch]$RegenerateStage2Base,
    [switch]$RegenerateAnalogBase,
    [switch]$RegenerateRankAwareBase,
    [switch]$RegenerateVectorAnenSource,
    [switch]$RegenerateModelStages,
    [switch]$StrictNoLockedStationRefine,
    [switch]$SkipBacktest
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "[e2e] NS surface d14 selective center candidate"
Write-Host "[e2e] RegenerateBase=$RegenerateBase RegenerateStationGateBase=$RegenerateStationGateBase RegenerateStage2Base=$RegenerateStage2Base RegenerateAnalogBase=$RegenerateAnalogBase RegenerateRankAwareBase=$RegenerateRankAwareBase RegenerateModelStages=$RegenerateModelStages StrictNoLockedStationRefine=$StrictNoLockedStationRefine SkipBacktest=$SkipBacktest"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_p7dir_mosres.csv"
if ($RegenerateBase -or -not (Test-Path $BaseCsv)) {
    Write-Host "[e2e] regenerating current base first"
    $baseScript = Join-Path $Root "run_ns_p7dir_mosres_e2e.ps1"
    $baseArgs = @()
    if ($RegenerateStationGateBase) { $baseArgs += "-RegenerateStationGateBase" }
    if ($RegenerateStage2Base) { $baseArgs += "-RegenerateStage2Base" }
    if ($RegenerateAnalogBase) { $baseArgs += "-RegenerateAnalogBase" }
    if ($RegenerateRankAwareBase) { $baseArgs += "-RegenerateRankAwareBase" }
    if ($RegenerateVectorAnenSource) { $baseArgs += "-RegenerateVectorAnenSource" }
    if ($RegenerateModelStages) { $baseArgs += "-RegenerateModelStages" }
    if ($StrictNoLockedStationRefine) { $baseArgs += "-StrictNoLockedStationRefine" }
    if ($SkipBacktest) { $baseArgs += "-SkipBacktest" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $baseScript @baseArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Base regeneration failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $BaseCsv)) {
    throw "Base CSV was not generated: $BaseCsv"
}

Write-Host "[e2e] building gated NS surface d14 selective center candidate"
& $Python (Join-Path $Root "build_ns_surface_d14_selective_center_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "Candidate generation failed with exit code $LASTEXITCODE"
}

$ZipPath = Join-Path $Root "runs\v6_pressure_speed\sub_ns_sfc14_sel.zip"
$Manifest = Join-Path $Root "runs\v6_pressure_speed\manifest_ns_sfc14_selective_center.json"
if (Test-Path $ZipPath) {
    if ((Split-Path -Leaf $ZipPath).Length -ge 64) {
        throw "Zip filename is too long for Codabench: $(Split-Path -Leaf $ZipPath)"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $names = @($zip.Entries | ForEach-Object { $_.FullName })
        if ($names.Count -ne 1 -or $names[0] -ne "predictions.csv") {
            throw "Zip must contain exactly root predictions.csv; found: $($names -join ', ')"
        }
    } finally {
        $zip.Dispose()
    }
    Write-Host "[e2e] generated $ZipPath"
} elseif (Test-Path $Manifest) {
    Write-Host "[e2e] no zip emitted; see gate manifest $Manifest"
} else {
    throw "Neither submission zip nor manifest was generated"
}
