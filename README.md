# Sea Winds Final Submission Folder

## Data Placement

Put the official Phase 1 dataset under:

`data`

Any one of these layouts is accepted:

```text
data/
  phase1_dataset/
    train/
    inference/
    scoring/
```

```text
data/
  train/
  inference/
  scoring/
```

```text
data/
  phase1_dataset.zip
```

The folder already includes the official starter-kit helper modules used by the pipeline:

`official_phase1/utils.py`

`official_phase1/feature_engineering.py`

## How To Run

From this folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_generate_final_submission.ps1
```

Optional arguments:

```powershell
-DataDir "D:\path\to\phase1_data"
-OutputDir "D:\path\to\output"
-SkipVenvSetup
-KeepGeneratedArtifacts
```

By default, the script creates `.venv`, installs `requirements.txt`, reads data from `data`, writes intermediate artifacts to `runs/v6_pressure_speed`, and copies the final ZIP to `final_submission_output/final_submission.zip`.

## Final Output

Primary final submission:

`final_submission_output/final_submission.zip`

Internal generated candidate:

`runs/v6_pressure_speed/sub_nsp1cdsel_v1.zip`

Expected public score for this selected artifact:

`primary_score = 1.414470`

Expected CSV SHA256 when regenerated with the same model/library behavior:

`be6eb6860bed47d31ca618552f71e53189e26ef7e981fa7ae17ec6efa8de06ad`

The ZIP is validated to contain exactly one root file named `predictions.csv`.

## Pipeline Summary

The runner rebuilds the full chain from official data:

1. Current-best rg90 base.
2. Feature-rich new-signal row cache.
3. Direction error-width new-signal branch.
4. Grid-long direction row cache.
5. ECS surface d14 direction push.
6. Final circular d1 selector.

All generated data stays inside this submission folder unless `-DataDir` or `-OutputDir` is provided.

