from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_nspos_b75.csv"
OUT_CSV = WORK / "pred_biggap_wcal.csv"
OUT_ZIP = WORK / "sub_biggap_wcal.zip"
MANIFEST = WORK / "manifest_biggap_wcal.json"
SPEED_CV = WORK / "cv_speed_width_scale_anchor.csv"
STATION_DIR_CV = WORK / "robust_station_direction_backtest_summary.csv"

COLS = [
    "type",
    "window",
    "region",
    "latitude",
    "longitude",
    "station",
    "horizon",
    "hour",
    "level",
    "q05",
    "q50",
    "q95",
    "dir_05",
    "dir_50",
    "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]
PRESSURE_LEVELS = {"1000", "925", "850", "700", "500"}
SURFACE_LEVELS = {"10m", "100m"}

# These blocks are selected from official-data backtests only.  They keep the
# forecast center fixed and recalibrate the interval shape/width.
SPEED_BLOCKS = [
    {
        "name": "ns_pressure_d14_width_asym",
        "region": "north_sea",
        "group": "pressure",
        "horizon": 14,
        "levels": PRESSURE_LEVELS,
        "lo_scale": 0.85,
        "hi_scale": 1.30,
        "anchor_cv_gain": 1.172199707596448,
    },
    {
        "name": "ecs_surface_d7_width_asym",
        "region": "east_china_sea",
        "group": "surface",
        "horizon": 7,
        "levels": SURFACE_LEVELS,
        "lo_scale": 0.95,
        "hi_scale": 0.75,
        "anchor_cv_gain": 0.34499023921531524,
    },
    {
        "name": "ecs_pressure_d7_width_asym",
        "region": "east_china_sea",
        "group": "pressure",
        "horizon": 7,
        "levels": PRESSURE_LEVELS,
        "lo_scale": 0.95,
        "hi_scale": 0.85,
        "anchor_cv_gain": 0.27782969755561915,
    },
    {
        "name": "ns_surface_d14_width_asym",
        "region": "north_sea",
        "group": "surface",
        "horizon": 14,
        "levels": SURFACE_LEVELS,
        "lo_scale": 0.95,
        "hi_scale": 1.05,
        "anchor_cv_gain": 0.07717423273027535,
    },
    {
        "name": "ecs_surface_d14_width_asym",
        "region": "east_china_sea",
        "group": "surface",
        "horizon": 14,
        "levels": SURFACE_LEVELS,
        "lo_scale": 0.95,
        "hi_scale": 1.05,
        "anchor_cv_gain": 0.07722995813452016,
    },
]

STATION_DIR_BLOCKS = [
    {
        "name": "ns_station_d7_monthclim_width",
        "region": "north_sea",
        "horizon": 7,
        "half_width": 102.5,
        "cv_candidate": "month_clim",
        "cv_score_mean": 284.4953728361347,
        "cv_score_max": 288.9931682137421,
    },
    {
        "name": "ns_station_d14_monthclim_width",
        "region": "north_sea",
        "horizon": 14,
        "half_width": 115.0,
        "cv_candidate": "month_clim",
        "cv_score_mean": 282.30573472071507,
        "cv_score_max": 311.85862937483245,
    },
    {
        "name": "ecs_station_d14_monthclim_width",
        "region": "east_china_sea",
        "horizon": 14,
        "half_width": 107.5,
        "cv_candidate": "month_clim",
        "cv_score_mean": 271.7475626217482,
        "cv_score_max": 273.0771505635663,
    },
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
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


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def row_diff(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def apply_speed_blocks(df: pd.DataFrame) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for block in SPEED_BLOCKS:
        mask = (
            df["type"].eq("grid")
            & df["region"].eq(str(block["region"]))
            & df["horizon"].eq(int(block["horizon"]))
            & df["level"].isin(block["levels"])
        )
        idx = df.index[mask]
        if len(idx) == 0:
            raise SystemExit(f"Speed block matched no rows: {block['name']}")
        old_lo = pd.to_numeric(df.loc[idx, "q05"], errors="coerce").to_numpy(dtype="float64")
        mid = pd.to_numeric(df.loc[idx, "q50"], errors="coerce").to_numpy(dtype="float64")
        old_hi = pd.to_numeric(df.loc[idx, "q95"], errors="coerce").to_numpy(dtype="float64")
        lo = np.maximum(0.0, mid - float(block["lo_scale"]) * (mid - old_lo))
        hi = mid + float(block["hi_scale"]) * (old_hi - mid)
        df.loc[idx, "q05"] = lo
        df.loc[idx, "q95"] = np.maximum(hi, mid)
        reports.append(
            {
                "name": block["name"],
                "rows": int(len(idx)),
                "region": block["region"],
                "group": block["group"],
                "horizon": int(block["horizon"]),
                "lo_scale": float(block["lo_scale"]),
                "hi_scale": float(block["hi_scale"]),
                "anchor_cv_gain": float(block["anchor_cv_gain"]),
                "old_half_width_mean": float(np.nanmean((old_hi - old_lo) / 2.0)),
                "new_half_width_mean": float(np.nanmean((np.maximum(hi, mid) - lo) / 2.0)),
            }
        )
    return reports


def apply_station_direction_blocks(df: pd.DataFrame) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for block in STATION_DIR_BLOCKS:
        mask = (
            df["type"].eq("station")
            & df["region"].eq(str(block["region"]))
            & df["horizon"].eq(int(block["horizon"]))
        )
        idx = df.index[mask]
        if len(idx) == 0:
            raise SystemExit(f"Station direction block matched no rows: {block['name']}")
        center = pd.to_numeric(df.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        old_hw = ((pd.to_numeric(df.loc[idx, "dir_95"], errors="coerce").to_numpy(dtype="float64")
                   - pd.to_numeric(df.loc[idx, "dir_05"], errors="coerce").to_numpy(dtype="float64")) % 360.0) / 2.0
        hw = float(block["half_width"])
        df.loc[idx, "dir_05"] = (center - hw) % 360.0
        df.loc[idx, "dir_95"] = (center + hw) % 360.0
        reports.append(
            {
                "name": block["name"],
                "rows": int(len(idx)),
                "region": block["region"],
                "horizon": int(block["horizon"]),
                "old_half_width_mean": float(np.nanmean(old_hw)),
                "new_half_width": hw,
                "cv_candidate": block["cv_candidate"],
                "cv_score_mean": float(block["cv_score_mean"]),
                "cv_score_max": float(block["cv_score_max"]),
            }
        )
    return reports


def validate(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    for c in SPEED_COLS:
        after[c] = pd.to_numeric(after[c], errors="coerce").clip(lower=0).round(2)
    after["q05"] = after[["q05", "q50"]].min(axis=1).round(2)
    after["q95"] = after[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        after[c] = ((pd.to_numeric(after[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)

    grid = after["type"].eq("grid")
    type_counts = after["type"].value_counts(dropna=False).to_dict()
    missing_pred = int(after[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    missing_grid_key = int(after.loc[grid, ["window", "region", "latitude", "longitude", "horizon", "hour", "level"]].isna().any(axis=1).sum())
    missing_station_key = int(after.loc[~grid, ["window", "region", "station", "horizon", "hour"]].isna().any(axis=1).sum())
    bad_speed = int(((after["q05"] > after["q50"]) | (after["q50"] > after["q95"]) | (after[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((after[DIR_COLS] < 0) | (after[DIR_COLS] >= 360) | after[DIR_COLS].isna()).any(axis=1).sum())
    grid_dup = int(after.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(after.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    speed_changed = row_diff(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = row_diff(before, after, DIR_COLS, 1, circular=True)
    center_speed_changed = row_diff(before, after, ["q50"], 2, circular=False)
    center_dir_changed = row_diff(before, after, ["dir_50"], 1, circular=True)

    if len(after) != 3_448_800 or type_counts.get("grid") != 3_447_360 or type_counts.get("station") != 1_440:
        raise SystemExit(f"row/type count validation failed: rows={len(after)} counts={type_counts}")
    if missing_pred or missing_grid_key or missing_station_key or bad_speed or bad_dir or grid_dup or station_dup:
        raise SystemExit(
            f"content validation failed: missing_pred={missing_pred} "
            f"missing_grid_key={missing_grid_key} missing_station_key={missing_station_key} "
            f"bad_speed={bad_speed} bad_dir={bad_dir} grid_dup={grid_dup} station_dup={station_dup}"
        )
    if int(center_speed_changed.sum()) or int(center_dir_changed.sum()):
        raise SystemExit(
            f"center changed unexpectedly: speed_centers={int(center_speed_changed.sum())} "
            f"dir_centers={int(center_dir_changed.sum())}"
        )

    return {
        "rows": int(len(after)),
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "speed_interval_rows_changed": int(speed_changed.sum()),
        "direction_interval_rows_changed": int(dir_changed.sum()),
        "speed_center_rows_changed": int(center_speed_changed.sum()),
        "direction_center_rows_changed": int(center_dir_changed.sum()),
        "missing_prediction_rows": missing_pred,
        "missing_grid_key_rows": missing_grid_key,
        "missing_station_key_rows": missing_station_key,
        "bad_speed_rows": bad_speed,
        "bad_direction_rows": bad_dir,
        "grid_duplicate_keys": grid_dup,
        "station_duplicate_keys": station_dup,
    }


def write_zip(df: pd.DataFrame) -> None:
    print(f"Writing {OUT_CSV}", flush=True)
    df[COLS].to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    if OUT_ZIP.name.__len__() >= 64:
        raise SystemExit(f"zip filename too long: {OUT_ZIP.name}")
    print(f"zip={OUT_ZIP} size={OUT_ZIP.stat().st_size:,} uncompressed={info.file_size:,}", flush=True)


def write_manifest(payload: dict[str, object]) -> None:
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing base CSV: {BASE_CSV}. Run .\\run_seasonal_ns_positive_e2e.ps1 first.")
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    before = normalize(pd.read_csv(BASE_CSV, low_memory=False))
    after = before.copy()
    speed_reports = apply_speed_blocks(after)
    station_dir_reports = apply_station_direction_blocks(after)
    audit = validate(before, after)
    write_zip(after)
    manifest = {
        "status": "submission_written",
        "out_csv": str(OUT_CSV),
        "out_zip": str(OUT_ZIP),
        "zip_predictions_sha256": sha256_zip_member(OUT_ZIP),
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": sha256(BASE_CSV),
        "audit": audit,
        "speed_blocks": speed_reports,
        "station_direction_blocks": station_dir_reports,
        "cv_artifacts": {
            "speed_width_anchor_cv": str(SPEED_CV),
            "speed_width_anchor_cv_sha256": sha256(SPEED_CV),
            "station_direction_cv": str(STATION_DIR_CV),
            "station_direction_cv_sha256": sha256(STATION_DIR_CV),
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "future_target_data_used": False,
            "notes": [
                "All changes are interval recalibrations around centers generated by the end-to-end official-data pipeline.",
                "Speed scales were selected from official historical anchor backtests.",
                "Station direction half-widths were selected from official historical station-direction CV.",
                "The builder does not read evaluation labels or any external dataset.",
            ],
        },
        "code_hashes": {
            "build_biggap_width_cal_candidate.py": sha256(Path(__file__).resolve()),
            "run_biggap_width_cal_e2e.ps1": sha256(ROOT / "run_biggap_width_cal_e2e.ps1"),
            "run_seasonal_ns_positive_e2e.ps1": sha256(ROOT / "run_seasonal_ns_positive_e2e.ps1"),
        },
    }
    write_manifest(manifest)
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True), flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
