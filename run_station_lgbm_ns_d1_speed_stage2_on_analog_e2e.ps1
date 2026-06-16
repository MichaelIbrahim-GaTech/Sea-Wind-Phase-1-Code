param(
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
$analogBase = Join-Path $workDir "predictions_ns_surface_d14_speed_analog_mos_v1_compact.csv"
$finalZip = Join-Path $workDir "submission_station_lgbm_ns_d1_speed_stage2_on_analog_compact.zip"
$manifest = Join-Path $workDir "station_lgbm_ns_d1_speed_stage2_on_analog_manifest.json"

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

if ($RegenerateAnalogBase -or -not (Test-Path -LiteralPath $analogBase)) {
    Invoke-Step "Rebuild NS surface d14 speed analog MOS base" {
        $analogScript = Join-Path $projectRoot "run_ns_surface_d14_speed_analog_mos_v1_e2e.ps1"
        $analogArgs = @()
        if ($script:RegenerateRankAwareBase) { $analogArgs += "-RegenerateRankAwareBase" }
        if ($script:RegenerateVectorAnenSource) { $analogArgs += "-RegenerateVectorAnenSource" }
        if ($script:RegenerateModelStages) { $analogArgs += "-RegenerateModelStages" }
        if ($script:StrictNoLockedStationRefine) { $analogArgs += "-StrictNoLockedStationRefine" }
        if ($script:SkipBacktest) { $analogArgs += "-SkipBacktest" }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $analogScript @analogArgs
    }
} else {
    Write-Host "Using existing analog MOS base: $analogBase"
}

Invoke-Step "Build NS station d1 speed stage2 LGBM candidate" {
    & $venvPython (Join-Path $projectRoot "build_station_lgbm_ns_d1_speed_stage2_on_analog_candidate.py")
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
