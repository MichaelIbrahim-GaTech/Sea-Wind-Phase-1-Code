param(
    [switch]$RegenerateStage2Base,
    [switch]$RegenerateAnalogBase,
    [switch]$RegenerateRankAwareBase,
    [switch]$RegenerateVectorAnenSource,
    [switch]$RegenerateModelStages,
    [switch]$StrictNoLockedStationRefine,
    [switch]$SkipBacktest
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"
$stage2Base = Join-Path $workDir "predictions_station_lgbm_ns_d1_speed_stage2_on_analog_compact.csv"
$finalZip = Join-Path $workDir "submission_station_uvres_ns_d1_dir_gate_on_stage2_compact.zip"
$manifest = Join-Path $workDir "station_uvres_ns_d1_dir_gate_manifest.json"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing Python environment: $venvPython"
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    $start = Get-Date
    $global:LASTEXITCODE = 0
    & $Body
    if (-not $?) {
        throw "Step failed: $Name"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name with exit code $LASTEXITCODE"
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("=== done: {0} ({1:n1}s) ===" -f $Name, $elapsed.TotalSeconds)
}

if ($RegenerateStage2Base -or -not (Test-Path -LiteralPath $stage2Base)) {
    Invoke-Step "Rebuild stage2 NS station d1 speed base" {
        $stage2Script = Join-Path $projectRoot "run_station_lgbm_ns_d1_speed_stage2_on_analog_e2e.ps1"
        $stage2Args = @()
        if ($script:RegenerateAnalogBase) { $stage2Args += "-RegenerateAnalogBase" }
        if ($script:RegenerateRankAwareBase) { $stage2Args += "-RegenerateRankAwareBase" }
        if ($script:RegenerateVectorAnenSource) { $stage2Args += "-RegenerateVectorAnenSource" }
        if ($script:RegenerateModelStages) { $stage2Args += "-RegenerateModelStages" }
        if ($script:StrictNoLockedStationRefine) { $stage2Args += "-StrictNoLockedStationRefine" }
        if ($script:SkipBacktest) { $stage2Args += "-SkipBacktest" }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $stage2Script @stage2Args
    }
} else {
    Write-Host "Using existing stage2 base: $stage2Base"
}

Invoke-Step "Build gated NS station d1 direction u/v MOS candidate" {
    & $venvPython (Join-Path $projectRoot "build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py")
}

if (-not (Test-Path -LiteralPath $finalZip)) {
    throw "Final zip was not created: $finalZip"
}
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Manifest was not created: $manifest"
}

Write-Host ""
Write-Host "Final submission: $finalZip"
Write-Host "Manifest: $manifest"
Get-Item $finalZip, $manifest | Select-Object FullName, Length, LastWriteTime
