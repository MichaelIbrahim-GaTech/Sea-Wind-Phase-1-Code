from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]

CODE_FILES = [
    "audit_final_submission.py",
    "run_final_submission_e2e.ps1",
    "run_public_positive_fullrefit_hybrid_e2e.ps1",
    "build_pressurefix_station_long_calibrated.py",
    "build_broad_grid_dir_candidate.py",
    "build_public_positive_fullrefit_hybrid_candidate.py",
    "build_station_lgbm_ecs_d1_direction_cv_candidate.py",
    "station_cv_mos_analog_framework.py",
    "sea_winds_solution_ephemeral_v6_pressure_speed.py",
    "compact_predictions_zip.py",
]

ARTIFACT_FILES = [
    "runs/v6_pressure_speed/baseline_station_refine_hybrid/predictions_station_refine_hybrid_bestknown.csv",
    "runs/v6_pressure_speed/predictions_station_refine_hybrid_bestknown.csv",
    "runs/v6_pressure_speed/predictions_direction_all_station_refine_compact.csv",
    "runs/v6_pressure_speed/predictions_proper_full_refit_v1_nsdirpost_compact.csv",
    "runs/v6_pressure_speed/predictions_public_positive_fullrefit_hybrid_compact.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_member(zip_path: Path, member: str) -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def read_submission_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if names != ["predictions.csv"]:
            raise SystemExit(f"zip must contain exactly root predictions.csv, got {names}")
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"zip test failed on member {bad}")
        with zf.open("predictions.csv") as f:
            return pd.read_csv(f, low_memory=False)[COLS]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    for c in ["latitude", "longitude"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    for c in SPEED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in DIR_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce") % 360.0
    return out


def validate(df: pd.DataFrame) -> dict:
    df = normalize(df)
    grid = df["type"].eq("grid")
    counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    grid_dup = int(df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(df.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    ok = (
        len(df) == 3_448_800
        and counts.get("grid") == 3_447_360
        and counts.get("station") == 1_440
        and bad_speed == 0
        and bad_dir == 0
        and missing == 0
        and grid_dup == 0
        and station_dup == 0
    )
    return {
        "ok": bool(ok),
        "rows": int(len(df)),
        "type_counts": {str(k): int(v) for k, v in counts.items()},
        "bad_speed": bad_speed,
        "bad_dir": bad_dir,
        "missing": missing,
        "grid_dup": grid_dup,
        "station_dup": station_dup,
    }


def diff_against_baseline(final_df: pd.DataFrame, baseline_csv: Path | None) -> dict | None:
    if baseline_csv is None or not baseline_csv.exists():
        return None
    base = normalize(pd.read_csv(baseline_csv, low_memory=False)[COLS])
    final = normalize(final_df)
    speed_changed = (base[SPEED_COLS].round(2).to_numpy() != final[SPEED_COLS].round(2).to_numpy()).any(axis=1)
    dir_changed = (base[DIR_COLS].round(1).to_numpy() != final[DIR_COLS].round(1).to_numpy()).any(axis=1)
    station = final["type"].eq("station")
    ecs_d1_station = station & final["region"].eq("east_china_sea") & final["horizon"].eq(1)
    return {
        "baseline_csv": str(baseline_csv),
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "station_rows_changed": int((station & dir_changed).sum()),
        "ecs_station_d1_direction_rows_changed": int((ecs_d1_station & dir_changed).sum()),
    }


def code_hashes(root: Path) -> dict:
    out = {}
    for rel in CODE_FILES:
        path = root / rel
        out[rel] = sha256(path) if path.exists() else None
    return out


def artifact_hashes(root: Path) -> dict:
    out = {}
    for rel in ARTIFACT_FILES:
        path = root / rel
        out[rel] = {
            "exists": path.exists(),
            "size": int(path.stat().st_size) if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--baseline-csv")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", default="locked-stage-e2e")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    zip_path = Path(args.zip)
    baseline = Path(args.baseline_csv) if args.baseline_csv else None
    final_df = read_submission_csv(zip_path)
    validation = validate(final_df)
    if not validation["ok"]:
        raise SystemExit(f"submission validation failed: {validation}")
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo("predictions.csv")
    manifest = {
        "mode": args.mode,
        "submission_zip": str(zip_path),
        "zip_size": int(zip_path.stat().st_size),
        "zip_sha256": sha256(zip_path),
        "internal_csv_name": "predictions.csv",
        "internal_csv_size": int(info.file_size),
        "internal_csv_sha256": sha256_zip_member(zip_path, "predictions.csv"),
        "validation": validation,
        "delta_vs_public_positive_baseline": diff_against_baseline(final_df, baseline),
        "public_reference": {
            "submitted_zip": "submission_station_lgbm_ecs_d1_dir_cv_compact.zip",
            "reported_primary_score": 1.460484,
            "reported_changed_metric": {
                "name": "dir_stations_d1_ecs",
                "previous": 230.7892,
                "new": 230.5785,
            },
        },
        "compliance": {
            "official_dataset_root": "runs/v6_pressure_speed/phase1_dataset",
            "external_training_data_used": False,
            "noncompliant_external_era5_used": False,
            "evaluation_targets_used_for_training": False,
        },
        "stage_artifact_hashes": artifact_hashes(root),
        "code_hashes": code_hashes(root),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
