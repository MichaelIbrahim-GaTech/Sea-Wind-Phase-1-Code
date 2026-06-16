from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
TRAIN = DATA / "train"

BASE_CSV = WORK / "predictions_rankaware_nsdir_stationmos_v1_compact.csv"
OUT_CSV = WORK / "predictions_ns_surface_d14_speed_analog_mos_v1_compact.csv"
OUT_ZIP = WORK / "submission_ns_surface_d14_speed_analog_mos_v1_compact.zip"
MANIFEST = WORK / "ns_surface_d14_speed_analog_mos_v1_manifest.json"

LEVELS = ("10m", "100m")
HOURS = (0, 6, 12, 18)
HORIZON = 14
HALF_WINDOW_DAYS = 45
TRAIN_YEARS = (2019, 2020, 2021)

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS


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


def load_surface_cube() -> tuple[pd.DataFrame, dict[pd.Timestamp, int], dict[str, np.ndarray]]:
    cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
    df = pd.read_parquet(TRAIN / "reanalysis_north_sea_6h.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df = df.sort_values(["time", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
    times = pd.Series(df["time"].unique()).sort_values().map(pd.Timestamp).to_list()
    n_times = len(times)
    n_grid = int(len(df) // n_times)
    latlon = df.loc[: n_grid - 1, ["latitude", "longitude"]].reset_index(drop=True)
    speed = {
        "10m": np.sqrt(df["u10"].to_numpy(dtype="float64") ** 2 + df["v10"].to_numpy(dtype="float64") ** 2).reshape(n_times, n_grid),
        "100m": np.sqrt(df["u100"].to_numpy(dtype="float64") ** 2 + df["v100"].to_numpy(dtype="float64") ** 2).reshape(n_times, n_grid),
    }
    return latlon, {t: i for i, t in enumerate(times)}, speed


def candidate_times(target: pd.Timestamp) -> list[pd.Timestamp]:
    out = []
    for year in TRAIN_YEARS:
        center = pd.Timestamp(year=year, month=target.month, day=target.day, hour=target.hour)
        for offset in range(-HALF_WINDOW_DAYS, HALF_WINDOW_DAYS + 1):
            out.append(center + pd.Timedelta(days=offset))
    return out


def analog_quantiles(
    speed: dict[str, np.ndarray],
    time_to_idx: dict[pd.Timestamp, int],
    target: pd.Timestamp,
    level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = [time_to_idx[t] for t in candidate_times(target) if t in time_to_idx]
    if len(idx) < 60:
        raise RuntimeError(f"not enough analog samples for {target} {level}: {len(idx)}")
    vals = speed[level][idx, :]
    return (
        np.nanquantile(vals, 0.05, axis=0),
        np.nanquantile(vals, 0.50, axis=0),
        np.nanquantile(vals, 0.95, axis=0),
    )


def inference_origin(window: int) -> pd.Timestamp:
    meta = json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text())
    return pd.Timestamp(meta["context_end"])


def build_analog_predictions() -> pd.DataFrame:
    print("Loading official North Sea surface reanalysis cube", flush=True)
    latlon, time_to_idx, speed = load_surface_cube()
    rows = []
    for window in range(1, 9):
        origin = inference_origin(window)
        for hour in HOURS:
            target = origin + pd.Timedelta(days=HORIZON, hours=hour)
            for level in LEVELS:
                q05, q50, q95 = analog_quantiles(speed, time_to_idx, target, level)
                part = latlon.copy()
                part["window"] = window
                part["region"] = "north_sea"
                part["horizon"] = HORIZON
                part["hour"] = hour
                part["level"] = level
                part["q05_analog"] = q05
                part["q50_analog"] = q50
                part["q95_analog"] = q95
                rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["window", "region", "latitude", "longitude", "horizon", "hour", "level"])


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if set(cols) == set(DIR_COLS):
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def validate_delta(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    changed_speed = rows_changed(before, after, SPEED_COLS, 2)
    changed_dir = rows_changed(before, after, DIR_COLS, 1)
    target = (
        after["type"].eq("grid")
        & after["region"].eq("north_sea")
        & after["level"].isin(LEVELS)
        & after["horizon"].eq(HORIZON)
    ).to_numpy()
    if int(changed_dir.sum()) != 0:
        raise SystemExit(f"unexpected direction delta: {int(changed_dir.sum())}")
    outside = changed_speed & ~target
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected speed delta outside NS surface d14: {int(outside.sum())}")
    return {
        "vs_rankaware_base": {
            "baseline_csv": str(BASE_CSV),
            "speed_rows_changed": int(changed_speed.sum()),
            "direction_rows_changed": int(changed_dir.sum()),
            "target_rows": int(target.sum()),
            "target_rows_unchanged_after_rounding": int((target & ~changed_speed).sum()),
        }
    }


def write_manifest(final: pd.DataFrame, delta: dict[str, object]) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")
    summary_path = WORK / "ns_surface_d14_speed_analog_backtest_summary.csv"
    by_year_path = WORK / "ns_surface_d14_speed_analog_backtest_by_year.csv"
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
            "ns_surface_d14_speed_rows": 164160,
            "levels": list(LEVELS),
            "hours": list(HOURS),
            "half_window_days": HALF_WINDOW_DAYS,
            "train_years": list(TRAIN_YEARS),
        },
        "delta": delta,
        "rank_aware_targets": ["WS NS Surface d14"],
        "validation_rationale": {
            "method": "calendar analog MOS from official North Sea surface reanalysis",
            "by_year_csv": str(by_year_path) if by_year_path.exists() else None,
            "summary_csv": str(summary_path) if summary_path.exists() else None,
            "selected_half_window_days": HALF_WINDOW_DAYS,
            "rolling_backtest_mean_score_for_45d": 13.742846,
            "rolling_backtest_max_score_for_45d": 14.068439,
        },
        "compliance": {
            "official_dataset_root": "runs/v6_pressure_speed/phase1_dataset",
            "external_training_data_used": False,
            "noncompliant_external_era5_used": False,
            "evaluation_targets_used_for_training": False,
            "notes": [
                "Uses only official 2019-2021 North Sea surface reanalysis to form calendar analog distributions.",
                "Final 2022 inference uses historical years only; no evaluation targets are read.",
                "No missing-data imputation is applied to inference meteorological fields.",
            ],
        },
        "artifact_hashes": {
            str(BASE_CSV): {"size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
            str(summary_path): {"size": int(summary_path.stat().st_size), "sha256": sha256(summary_path)} if summary_path.exists() else None,
        },
        "code_hashes": {
            "build_ns_surface_d14_speed_analog_mos_v1_candidate.py": sha256(Path(__file__).resolve()),
            "ns_surface_d14_speed_analog_backtest.py": sha256(ROOT / "ns_surface_d14_speed_analog_backtest.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_rankaware_nsdir_stationmos_v1_e2e.ps1 first.")
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = E2E.normalize_for_assembly(pd.read_csv(BASE_CSV, low_memory=False)[COLS])
    before = df.copy()
    analog = build_analog_predictions()
    mask = (
        df["type"].eq("grid")
        & df["region"].eq("north_sea")
        & df["level"].isin(LEVELS)
        & df["horizon"].eq(HORIZON)
    )
    lookup = df.loc[mask].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]]
    merged = lookup.merge(
        analog,
        on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
        how="left",
        validate="one_to_one",
    )
    if merged[["q05_analog", "q50_analog", "q95_analog"]].isna().any().any():
        raise SystemExit("missing analog quantiles for final patch")
    idx = merged["index"].to_numpy(dtype="int64")
    df.loc[idx, "q05"] = merged["q05_analog"].to_numpy(dtype="float64")
    df.loc[idx, "q50"] = merged["q50_analog"].to_numpy(dtype="float64")
    df.loc[idx, "q95"] = merged["q95_analog"].to_numpy(dtype="float64")
    print(f"Patched NS surface d14 speed rows: {len(idx):,}", flush=True)

    final = E2E.validate_final(df)
    delta = validate_delta(before, final)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, delta)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
