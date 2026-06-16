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

Write-Host "[e2e] RegenerateBase=$RegenerateBase RegenerateStationGateBase=$RegenerateStationGateBase RegenerateStage2Base=$RegenerateStage2Base RegenerateAnalogBase=$RegenerateAnalogBase RegenerateRankAwareBase=$RegenerateRankAwareBase RegenerateModelStages=$RegenerateModelStages StrictNoLockedStationRefine=$StrictNoLockedStationRefine SkipBacktest=$SkipBacktest"

$BaseCsv = Join-Path $Root "runs\v6_pressure_speed\pred_ns_d1dir_ensw_gate.csv"
if ($RegenerateBase -or -not (Test-Path $BaseCsv)) {
    Write-Host "[e2e] regenerating current accepted base with NS d1 direction ensemble-width gate"
    $baseScript = Join-Path $Root "run_ns_d1dir_enswidth_gate_e2e.ps1"
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

Write-Host "[e2e] building strict NS station d1 speed calibration-gate candidate"
& $Python (Join-Path $Root "build_ns_station_d1_speed_calib_gate_candidate.py")
if ($LASTEXITCODE -ne 0) {
    throw "Candidate generation failed with exit code $LASTEXITCODE"
}

$ZipPath = Join-Path $Root "runs\v6_pressure_speed\sub_ns_d1spd_calib_gate.zip"
if (-not (Test-Path $ZipPath)) {
    throw "Expected submission zip was not generated: $ZipPath"
}

Write-Host "[e2e] generated $ZipPath"
