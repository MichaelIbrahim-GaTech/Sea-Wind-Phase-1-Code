from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E


WORK = Path("runs/v6_pressure_speed")
FEATURES = WORK / "phase1_dataset" / "features"

BASE_CSV = WORK / "predictions_rankaware_nsdir_stationmos_v1_compact.csv"
OUT_CSV = WORK / "predictions_rankaware_pressure_d14_hresd10_v1_compact.csv"
OUT_ZIP = WORK / "submission_rankaware_pressure_d14_hresd10_v1_compact.zip"
MANIFEST = WORK / "rankaware_pressure_d14_hresd10_v1_manifest.json"

HOURS = (0, 6, 12, 18)
HALF_WIDTH = 155.0
PRESSURE_LEVELS = E2E.PRESSURE_LEVELS
DIR_COLS = E2E.DIR_COLS
SPEED_COLS = E2E.SPEED_COLS
COLS = E2E.COLS
KEYS = E2E.KEYS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_member(zip_path: Path, member: str = "predictions.csv") -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if set(cols) == set(DIR_COLS):
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    left = left.to_numpy()
    right = right.to_numpy()
    return (left != right).any(axis=1)


def assert_same_keys(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    lhs = a[KEYS].fillna("__NA__").astype(str)
    rhs = b[KEYS].fillna("__NA__").astype(str)
    if not bool((lhs.values == rhs.values).all()):
        raise SystemExit(f"{label}: key/order mismatch")


def load_hres_d10_pressure_ns() -> pd.DataFrame:
    rows = []
    for window in range(1, 9):
        feature_path = FEATURES / f"inference_window_{window}_north_sea.parquet"
        cols = ["latitude", "longitude"]
        for level in PRESSURE_LEVELS:
            for hour in HOURS:
                cols.append(f"fcst_u_{level}_d10_h{hour}")
                cols.append(f"fcst_v_{level}_d10_h{hour}")
        feat = pd.read_parquet(feature_path, columns=cols)
        feat["latitude"] = feat["latitude"].astype(float).round(2)
        feat["longitude"] = feat["longitude"].astype(float).round(2)
        for level in PRESSURE_LEVELS:
            for hour in HOURS:
                u = pd.to_numeric(feat[f"fcst_u_{level}_d10_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                v = pd.to_numeric(feat[f"fcst_v_{level}_d10_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                part = feat[["latitude", "longitude"]].copy()
                part["window"] = window
                part["region"] = "north_sea"
                part["horizon"] = 14
                part["hour"] = hour
                part["level"] = level
                part["hres_dir"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
                rows.append(part)
    hres = pd.concat(rows, ignore_index=True)
    return hres.drop_duplicates(["window", "region", "latitude", "longitude", "horizon", "hour", "level"])


def apply_ns_pressure_d14_hresd10(df: pd.DataFrame) -> int:
    print("Loading provided North Sea HRES d10 pressure-vector directions", flush=True)
    hres = load_hres_d10_pressure_ns()
    mask = (
        df["type"].eq("grid")
        & df["region"].eq("north_sea")
        & df["level"].isin(PRESSURE_LEVELS)
        & df["horizon"].eq(14)
    )
    lookup = df.loc[mask].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]]
    merged = lookup.merge(
        hres,
        on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
        how="left",
        validate="one_to_one",
    )
    missing = int(merged["hres_dir"].isna().sum())
    if missing:
        raise SystemExit(f"missing HRES d10 pressure directions: {missing}")

    idx = merged["index"].to_numpy()
    center = merged["hres_dir"].to_numpy(dtype="float64") % 360.0
    df.loc[idx, "dir_50"] = center
    df.loc[idx, "dir_05"] = (center - HALF_WIDTH) % 360.0
    df.loc[idx, "dir_95"] = (center + HALF_WIDTH) % 360.0
    return int(len(idx))


def validate_delta(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    assert_same_keys(before, after, "rank-aware base")
    changed_speed = rows_changed(before, after, SPEED_COLS, 2)
    changed_dir = rows_changed(before, after, DIR_COLS, 1)
    target = (
        after["type"].eq("grid")
        & after["region"].eq("north_sea")
        & after["level"].isin(PRESSURE_LEVELS)
        & after["horizon"].eq(14)
    ).to_numpy()
    if int(changed_speed.sum()) != 0:
        raise SystemExit(f"unexpected speed delta vs rank-aware base: {int(changed_speed.sum())}")
    outside = changed_dir & ~target
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected direction delta outside NS pressure d14: {int(outside.sum())}")
    return {
        "vs_rankaware_base": {
            "baseline_csv": str(BASE_CSV),
            "speed_rows_changed": int(changed_speed.sum()),
            "direction_rows_changed": int(changed_dir.sum()),
            "target_rows": int(target.sum()),
            "target_rows_unchanged_after_rounding": int((target & ~changed_dir).sum()),
        }
    }


def write_manifest(final: pd.DataFrame, patch_rows: int, delta: dict[str, object]) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")

    manifest = {
        "submission": {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_name": "predictions.csv",
            "internal_csv_size": int(info.file_size),
            "internal_csv_sha256": sha256_zip_member(OUT_ZIP),
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
            "bad_speed": 0,
            "bad_dir": 0,
            "missing": 0,
            "grid_dup": 0,
            "station_dup": 0,
        },
        "component_counts": {
            "ns_pressure_d14_hresd10_rows": int(patch_rows),
            "ns_pressure_d14_half_width": HALF_WIDTH,
        },
        "delta": delta,
        "rank_aware_targets": [
            "Dir NS Pressure d14",
        ],
        "frozen_by_design": [
            "All speed rows",
            "All station rows",
            "Rank-aware v1 positive direction blocks: NS station d14, NS surface d7, NS pressure d7, ECS station d1",
            "ECS pressure d7/d14 and ECS surface d14, where current public metrics already beat the reference leader row",
        ],
        "validation_rationale": {
            "source": "runs/v6_pressure_speed/robust_hres_direction_backtest_summary.csv",
            "north_sea_pressure_d14_hres_by_level_score_mean": 333.99884030914274,
            "north_sea_pressure_d14_hres_by_level_score_max": 336.1596599570423,
            "north_sea_pressure_d14_hres_by_level_half_width_mean": 152.5,
            "selected_half_width": HALF_WIDTH,
            "public_feedback_avoid": [
                "The learned 700-level proxy for NS pressure d14 worsened publicly.",
                "Width-only calibration barely moved this dimension.",
            ],
        },
        "compliance": {
            "official_dataset_root": "runs/v6_pressure_speed/phase1_dataset",
            "external_training_data_used": False,
            "noncompliant_external_era5_used": False,
            "evaluation_targets_used_for_training": False,
            "notes": [
                "Uses only official provided inference HRES pressure fields for the prediction windows.",
                "No external weather/reanalysis data is read.",
                "No public evaluation targets or scorer ground truth are used.",
            ],
        },
        "artifact_hashes": {
            str(BASE_CSV): {"size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
            str(FEATURES): "directory of official provided inference feature parquet files",
        },
        "code_hashes": {
            "build_rankaware_pressure_d14_hresd10_v1_candidate.py": sha256(Path(__file__).resolve()),
            "build_rankaware_nsdir_stationmos_v1_candidate.py": sha256(Path("build_rankaware_nsdir_stationmos_v1_candidate.py")),
            "sea_winds_end_to_end_final.py": sha256(Path("sea_winds_end_to_end_final.py")),
            "robust_hres_direction_backtest.py": sha256(Path("robust_hres_direction_backtest.py")),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_rankaware_nsdir_stationmos_v1_e2e.ps1 first.")
    print(f"Reading rank-aware base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[COLS]
    df = E2E.normalize_for_assembly(df)
    before = df.copy()

    patch_rows = apply_ns_pressure_d14_hresd10(df)
    print(f"Patched North Sea pressure d14 direction rows: {patch_rows:,}; half_width={HALF_WIDTH}", flush=True)

    final = E2E.validate_final(df)
    delta = validate_delta(before, final)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, patch_rows, delta)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
