param(
    [ValidateSet("smoke", "pilot", "direction")]
    [string]$Mode = "pilot"
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing local Python environment at $Python"
}

switch ($Mode) {
    "smoke" {
        & $Python .\cv_mos_residual_framework.py `
            --blocks "north_sea:surface:7,east_china_sea:pressure:7" `
            --problems "direction" `
            --val-years "2021" `
            --train-combos 20000 `
            --val-grid-per-anchor 100 `
            --direction-estimators 80 `
            --tag "smoke"
    }
    "pilot" {
        & $Python .\cv_mos_residual_framework.py `
            --blocks "north_sea:surface:7,north_sea:pressure:7,east_china_sea:pressure:7,east_china_sea:surface:14" `
            --problems "speed,direction" `
            --val-years "2020,2021" `
            --train-combos 60000 `
            --val-grid-per-anchor 180 `
            --speed-estimators 180 `
            --direction-estimators 180 `
            --tag "pilot"
    }
    "direction" {
        & $Python .\cv_mos_residual_framework.py `
            --blocks "north_sea:surface:7,north_sea:pressure:7,east_china_sea:pressure:7,east_china_sea:surface:14" `
            --problems "direction" `
            --val-years "2020,2021" `
            --train-combos 70000 `
            --val-grid-per-anchor 200 `
            --direction-estimators 180 `
            --tag "direction_targeted"
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "CV MOS/residual framework failed with exit code $LASTEXITCODE"
}
