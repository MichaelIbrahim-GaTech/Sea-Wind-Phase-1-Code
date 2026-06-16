param(
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
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $projectRoot "runs\v6_pressure_speed"
$stationGateBase = Join-Path $workDir "predictions_station_uvres_ns_d1_dir_gate_on_stage2_compact.csv"
$finalZip = Join-Path $workDir "sub_ns_d1dir_ensw_gate.zip"
$manifest = Join-Path $workDir "manifest_ns_d1dir_ensw_gate.json"
$auditManifest = Join-Path $workDir "audit_ns_d1dir_ensw_gate.json"

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

if ($RegenerateStationGateBase -or -not (Test-Path -LiteralPath $stationGateBase)) {
    Invoke-Step "Rebuild station-gate base" {
        $baseScript = Join-Path $projectRoot "run_station_uvres_ns_d1_direction_gate_on_stage2_e2e.ps1"
        $baseArgs = @()
        if ($script:RegenerateStage2Base) { $baseArgs += "-RegenerateStage2Base" }
        if ($script:RegenerateAnalogBase) { $baseArgs += "-RegenerateAnalogBase" }
        if ($script:RegenerateRankAwareBase) { $baseArgs += "-RegenerateRankAwareBase" }
        if ($script:RegenerateVectorAnenSource) { $baseArgs += "-RegenerateVectorAnenSource" }
        if ($script:RegenerateModelStages) { $baseArgs += "-RegenerateModelStages" }
        if ($script:StrictNoLockedStationRefine) { $baseArgs += "-StrictNoLockedStationRefine" }
        if ($script:SkipBacktest) { $baseArgs += "-SkipBacktest" }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $baseScript @baseArgs
    }
} else {
    Write-Host "Using existing station-gate base: $stationGateBase"
}

Invoke-Step "Build gated NS station d1 direction ensemble-width candidate" {
    & $venvPython (Join-Path $projectRoot "build_ns_d1dir_enswidth_gate_candidate.py")
}

Invoke-Step "Audit final zip" {
    & $venvPython (Join-Path $projectRoot "audit_final_submission.py") `
        --zip $finalZip `
        --baseline-csv $stationGateBase `
        --manifest $auditManifest `
        --mode ns_d1dir_ensw_gate
}

if (-not (Test-Path -LiteralPath $finalZip)) {
    throw "Final zip was not created: $finalZip"
}
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Manifest was not created: $manifest"
}
if (-not (Test-Path -LiteralPath $auditManifest)) {
    throw "Audit manifest was not created: $auditManifest"
}

Write-Host ""
Write-Host "Final submission: $finalZip"
Write-Host "Manifest: $manifest"
Write-Host "Audit manifest: $auditManifest"
Get-Item $finalZip, $manifest, $auditManifest | Select-Object FullName, Length, LastWriteTime
