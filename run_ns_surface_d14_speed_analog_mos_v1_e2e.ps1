param(
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
$baseCsv = Join-Path $workDir "predictions_rankaware_nsdir_stationmos_v1_compact.csv"
$finalZip = Join-Path $workDir "submission_ns_surface_d14_speed_analog_mos_v1_compact.zip"
$manifest = Join-Path $workDir "ns_surface_d14_speed_analog_mos_v1_manifest.json"

if (-not (Test-Path $venvPython)) {
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
    & $Body
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("=== done: {0} ({1:n1}s) ===" -f $Name, $elapsed.TotalSeconds)
}

function Invoke-CommandStep {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments = @()
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    $start = Get-Date
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("=== done: {0} ({1:n1}s) ===" -f $Name, $elapsed.TotalSeconds)
}

if ($RegenerateRankAwareBase -or -not (Test-Path $baseCsv)) {
    $rankArgs = @()
    if ($RegenerateVectorAnenSource) {
        $rankArgs += "-RegenerateVectorAnenSource"
    }
    if ($RegenerateModelStages) {
        $rankArgs += "-RegenerateModelStages"
    }
    if ($StrictNoLockedStationRefine) {
        $rankArgs += "-StrictNoLockedStationRefine"
    }
    Invoke-Step "Rebuild rank-aware base" {
        $rankScript = Join-Path $projectRoot "run_rankaware_nsdir_stationmos_v1_e2e.ps1"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $rankScript @rankArgs
    }
} else {
    Write-Host "Using existing rank-aware base: $baseCsv"
}

if (-not $SkipBacktest) {
    Invoke-Step "Run official-data rolling analog backtest" {
        & $venvPython (Join-Path $projectRoot "ns_surface_d14_speed_analog_backtest.py")
    }
}

Invoke-Step "Build NS surface d14 speed analog MOS submission" {
    & $venvPython (Join-Path $projectRoot "build_ns_surface_d14_speed_analog_mos_v1_candidate.py")
}

if (-not (Test-Path $finalZip)) {
    throw "Final zip was not created: $finalZip"
}
if (-not (Test-Path $manifest)) {
    throw "Manifest was not created: $manifest"
}

Write-Host ""
Write-Host "Final submission: $finalZip"
Write-Host "Manifest: $manifest"
Get-Item $finalZip, $manifest | Select-Object FullName, Length, LastWriteTime
