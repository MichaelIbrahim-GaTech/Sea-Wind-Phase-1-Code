$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\build_circular_distribution_selective_d1_v1_candidate.py
if ($LASTEXITCODE -ne 0) {
    throw "build_circular_distribution_selective_d1_v1_candidate.py failed with exit code $LASTEXITCODE"
}
