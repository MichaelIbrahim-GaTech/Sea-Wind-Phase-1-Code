param(
    [string]$DataDir = "",
    [string]$OfficialPhase1Dir = "",
    [string]$OutputDir = "",
    [switch]$KeepGeneratedArtifacts,
    [switch]$AllowLockedStationRefine,
    [switch]$SkipBacktest,
    [switch]$SkipVenvSetup,
    [int]$GridPerAnchor = 450
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$WorkDir = Join-Path $Root "runs\v6_pressure_speed"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Root "final_submission_output"
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    $global:LASTEXITCODE = 0
    $start = Get-Date
    & $Command
    if (-not $?) {
        throw "$Name failed."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("=== done: {0} ({1:n1}s) ===" -f $Name, $elapsed.TotalSeconds)
}

function Resolve-OfficialPhase1Dir {
    param([string]$RequestedPath)

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $candidates += $RequestedPath
    }
    if (-not [string]::IsNullOrWhiteSpace($env:SEA_WINDS_OFFICIAL_PHASE1_DIR)) {
        $candidates += $env:SEA_WINDS_OFFICIAL_PHASE1_DIR
    }
    $candidates += (Join-Path $Root "official_phase1")
    $candidates += (Join-Path $Root "data\phase_1")
    $candidates += (Join-Path $Root "data\Hackathon-Sea-Winds-Predictions\phase_1")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw @"
Official starter-kit phase_1 modules were not found.

This submission folder should contain:
  official_phase1\utils.py
  official_phase1\feature_engineering.py
"@
}

function Resolve-DataDir {
    param([string]$RequestedPath)

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $candidates += $RequestedPath
    }
    if (-not [string]::IsNullOrWhiteSpace($env:SEA_WINDS_DATA_DIR)) {
        $candidates += $env:SEA_WINDS_DATA_DIR
    }
    $candidates += (Join-Path $Root "data")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Data directory was not found. Create a data directory beside this script and put the official Phase 1 data there."
}

function Test-Phase1DatasetInDataDir {
    param([string]$DataRoot)

    $roots = @(
        (Join-Path $DataRoot "phase1_dataset"),
        $DataRoot
    )
    try {
        $roots += @(Get-ChildItem -LiteralPath $DataRoot -Directory | ForEach-Object { $_.FullName })
    } catch {
    }

    foreach ($candidate in $roots) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (
            (Test-Path -LiteralPath (Join-Path $candidate "train") -PathType Container) -and
            (Test-Path -LiteralPath (Join-Path $candidate "inference") -PathType Container) -and
            (Test-Path -LiteralPath (Join-Path $candidate "scoring") -PathType Container)
        ) {
            return $true
        }
    }

    $zipPath = Join-Path $DataRoot "phase1_dataset.zip"
    if (Test-Path -LiteralPath $zipPath -PathType Leaf) {
        return $true
    }

    return $false
}

function Ensure-PythonEnvironment {
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }

    if ($SkipVenvSetup) {
        Write-Host "[env] .venv not found; using python from PATH because -SkipVenvSetup was provided."
        return "python"
    }

    Invoke-Step "Create local Python environment" {
        & python -m venv (Join-Path $Root ".venv")
    }
    Invoke-Step "Install Python requirements" {
        & $venvPython -m pip install --upgrade pip
        & $venvPython -m pip install -r (Join-Path $Root "requirements.txt")
    }
    return $venvPython
}

function Clear-GeneratedArtifacts {
    if (-not (Test-Path -LiteralPath $WorkDir)) {
        New-Item -ItemType Directory -Path $WorkDir | Out-Null
        return
    }

    $extensions = @(".csv", ".zip", ".json", ".md", ".parquet", ".pkl", ".joblib", ".txt")
    Get-ChildItem -LiteralPath $WorkDir -File |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force
        }
}

function Assert-ZipLayout {
    param([string]$ZipPath)

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $names = @($zip.Entries | ForEach-Object { $_.FullName })
        if ($names.Count -ne 1 -or $names[0] -ne "predictions.csv") {
            throw "Submission zip must contain exactly one root predictions.csv; found: $($names -join ', ')"
        }
    } finally {
        $zip.Dispose()
    }
}

$dataRoot = Resolve-DataDir -RequestedPath $DataDir
$officialDir = Resolve-OfficialPhase1Dir -RequestedPath $OfficialPhase1Dir
if (-not (Test-Phase1DatasetInDataDir -DataRoot $dataRoot)) {
    throw @"
No official Phase 1 dataset was found under:
  $dataRoot

Put the data in one of these layouts:
  data\phase1_dataset\train, data\phase1_dataset\inference, data\phase1_dataset\scoring
  data\train, data\inference, data\scoring
  data\phase1_dataset.zip
"@
}

New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:SEA_WINDS_WORKDIR = $WorkDir
$env:SEA_WINDS_DATA_DIR = $dataRoot
$env:SEA_WINDS_OFFICIAL_PHASE1_DIR = $officialDir
$env:SEA_WINDS_USE_PERSISTENT_CACHE = "1"
$env:SEA_WINDS_PERSIST_DIR = $dataRoot
$env:SEA_WINDS_PERSIST_EXTRACTED = "1"
$env:SEA_WINDS_KEEP_ZIP = "1"
$env:SEA_WINDS_LOW_RAM = "1"
$env:SEA_WINDS_FINALIZE_EXISTING = "0"

Write-Host "[setup] root:      $Root"
Write-Host "[setup] data:      $dataRoot"
Write-Host "[setup] official:  $officialDir"
Write-Host "[setup] workdir:   $WorkDir"
Write-Host "[setup] outputdir: $OutputDir"

if (-not $KeepGeneratedArtifacts) {
    Invoke-Step "Clean generated top-level run artifacts" {
        Clear-GeneratedArtifacts
    }
} else {
    Write-Host "[setup] keeping existing generated artifacts because -KeepGeneratedArtifacts was provided"
}

$Python = Ensure-PythonEnvironment
if ($Python -ne "python") {
    $pythonDir = Split-Path -Parent $Python
    $env:PATH = "$pythonDir;$env:PATH"
}
Write-Host "[setup] python:    $Python"

$regenArgs = @(
    "-RegenerateBase",
    "-RegenerateOrigBase",
    "-RegenerateSelectiveCenter",
    "-RegenerateStationGateBase",
    "-RegenerateStage2Base",
    "-RegenerateAnalogBase",
    "-RegenerateRankAwareBase",
    "-RegenerateVectorAnenSource",
    "-RegenerateModelStages",
    "-GridPerAnchor",
    "$GridPerAnchor"
)
if (-not $AllowLockedStationRefine) {
    $regenArgs += "-StrictNoLockedStationRefine"
}
if ($SkipBacktest) {
    $regenArgs += "-SkipBacktest"
} else {
    $regenArgs += "-RegenerateBacktest"
}

Invoke-Step "Build current-best rg90 base from official inputs" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_ns_sfc14_rg90_e2e.ps1") @regenArgs
}

Invoke-Step "Build feature-rich new-signal row cache" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_feature_rich_newsignal_v1_e2e.ps1")
}

Invoke-Step "Build direction error-width new-signal branch" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_dir_error_width_newsignal_v1_e2e.ps1")
}

Invoke-Step "Build grid-long direction row cache" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_dir_error_width_gridlong_v1_e2e.ps1")
}

Invoke-Step "Apply ECS surface d14 direction push" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_direrrw_ecss14push_v1_e2e.ps1")
}

Invoke-Step "Apply final circular d1 selector" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_circular_distribution_selective_d1_v1_e2e.ps1")
}

$finalCsv = Join-Path $WorkDir "pred_nsp1cdsel_v1.csv"
$finalZip = Join-Path $WorkDir "sub_nsp1cdsel_v1.zip"
if (-not (Test-Path -LiteralPath $finalCsv)) {
    throw "Final CSV was not generated: $finalCsv"
}
if (-not (Test-Path -LiteralPath $finalZip)) {
    throw "Final ZIP was not generated: $finalZip"
}
Assert-ZipLayout -ZipPath $finalZip

$copyZip = Join-Path $OutputDir "final_submission.zip"
Copy-Item -LiteralPath $finalZip -Destination $copyZip -Force

$csvHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalCsv).Hash.ToLowerInvariant()
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalZip).Hash.ToLowerInvariant()
$expectedCsvHash = "be6eb6860bed47d31ca618552f71e53189e26ef7e981fa7ae17ec6efa8de06ad"

Write-Host ""
Write-Host "Final generated submission:"
Write-Host "  zip:      $finalZip"
Write-Host "  copy:     $copyZip"
Write-Host "  csv:      $finalCsv"
Write-Host "  csv sha:  $csvHash"
Write-Host "  zip sha:  $zipHash"
Write-Host "  public-selected score: primary_score 1.414470 for sub_nsp1cdsel_v1.zip"
if ($csvHash -eq $expectedCsvHash) {
    Write-Host "  reproducibility: CSV hash matches the selected public artifact."
} else {
    Write-Warning "CSV hash differs from the selected public artifact. This usually means regenerated model stages or package versions differ; the artifact was still produced end-to-end from official inputs."
}
