from __future__ import annotations

import zipfile
import gc
from pathlib import Path

import pandas as pd


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "predictions_station_refine_hybrid_bestknown.csv"
DIR_ALL_CSV = WORK / "predictions_direction_all_station_refine_compact.csv"

PRESSURE_DIRALL_CSV = WORK / "predictions_pressure_dirall_on_bestknown_compact.csv"
PRESSURE_DIRALL_ZIP = WORK / "submission_pressure_dirall_on_bestknown_compact.zip"
PRESSURE_REVERT_CSV = WORK / "predictions_pressure_dirall_revert_ns_p14_compact.csv"
PRESSURE_REVERT_ZIP = WORK / "submission_pressure_dirall_revert_ns_p14_compact.zip"
OUT_CSV = WORK / "predictions_pressurefix_station_long_calibrated_compact.csv"
OUT_ZIP = WORK / "submission_pressurefix_station_long_calibrated_compact.zip"

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
KEYS = ["type", "window", "region", "latitude", "longitude", "station", "horizon", "hour", "level"]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]
PRESSURE_LEVELS = ["1000", "925", "850", "700", "500"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for col in ["window", "horizon", "hour"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("int64")
    for col in ["latitude", "longitude"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    for col in SPEED_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0).round(2)
    out["q05"] = out[["q05", "q50"]].min(axis=1).round(2)
    out["q95"] = out[["q95", "q50"]].max(axis=1).round(2)
    for col in DIR_COLS:
        out[col] = ((pd.to_numeric(out[col], errors="coerce") % 360.0).round(1) % 360.0).round(1)
    return out


def assert_same_keys(a: pd.DataFrame, b: pd.DataFrame) -> None:
    lhs = a[KEYS].fillna("__NA__").astype(str)
    rhs = b[KEYS].fillna("__NA__").astype(str)
    if not bool((lhs.values == rhs.values).all()):
        raise RuntimeError("stage files are not aligned on submission keys")


def validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = normalize(df)
    grid = out["type"].eq("grid")
    counts = out["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((out["q05"] > out["q50"]) | (out["q50"] > out["q95"]) | (out[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((out[DIR_COLS] < 0) | (out[DIR_COLS] >= 360) | out[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(out[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    grid_dup = int(out.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(out.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    print(
        f"{label}: rows={len(out):,}; counts={counts}; bad_speed={bad_speed}; "
        f"bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}",
        flush=True,
    )
    if len(out) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise RuntimeError(f"{label}: row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise RuntimeError(f"{label}: content validation failed")
    return out[COLS]


def write_stage(df: pd.DataFrame, csv_path: Path, zip_path: Path, label: str) -> None:
    final = validate(df, label)
    print(f"Writing {csv_path}", flush=True)
    final.to_csv(csv_path, index=False)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    print(f"  zip={zip_path}; names={names}; uncompressed={info.file_size:,}", flush=True)


def main() -> None:
    print(f"Reading base station-refine stage {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize(pd.read_csv(BASE_CSV, low_memory=False))
    print(f"Reading all-level direction stage {DIR_ALL_CSV} ({DIR_ALL_CSV.stat().st_size:,} bytes)", flush=True)
    dir_all = normalize(pd.read_csv(DIR_ALL_CSV, low_memory=False))
    assert_same_keys(base, dir_all)

    pressure_dirall = base.copy()
    pressure_mask = pressure_dirall["type"].eq("grid") & pressure_dirall["level"].isin(PRESSURE_LEVELS)
    pressure_dirall.loc[pressure_mask, DIR_COLS] = dir_all.loc[pressure_mask, DIR_COLS].to_numpy()
    print(f"pressure_dirall copied pressure-grid direction rows: {int(pressure_mask.sum()):,}", flush=True)
    write_stage(pressure_dirall, PRESSURE_DIRALL_CSV, PRESSURE_DIRALL_ZIP, "pressure_dirall_on_bestknown")

    del pressure_dirall
    gc.collect()

    pressure_revert = base.copy()
    pressure_revert.loc[pressure_mask, DIR_COLS] = dir_all.loc[pressure_mask, DIR_COLS].to_numpy()
    revert_mask = (
        pressure_revert["type"].eq("grid")
        & pressure_revert["region"].eq("north_sea")
        & pressure_revert["horizon"].eq(14)
        & pressure_revert["level"].isin(PRESSURE_LEVELS)
    )
    pressure_revert.loc[revert_mask, DIR_COLS] = base.loc[revert_mask, DIR_COLS].to_numpy()
    print(f"pressure_revert restored NS pressure d14 direction rows: {int(revert_mask.sum()):,}", flush=True)
    write_stage(pressure_revert, PRESSURE_REVERT_CSV, PRESSURE_REVERT_ZIP, "pressure_dirall_revert_ns_p14")

    pressurefix = pressure_revert
    station_policies = [
        ("east_china_sea", 14, 120.0),
        ("north_sea", 7, 135.0),
        ("north_sea", 14, 150.0),
    ]
    changed = 0
    for region, horizon, half_width in station_policies:
        mask = pressurefix["type"].eq("station") & pressurefix["region"].eq(region) & pressurefix["horizon"].eq(horizon)
        center = pd.to_numeric(dir_all.loc[mask, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        pressurefix.loc[mask, "dir_50"] = center
        pressurefix.loc[mask, "dir_05"] = (center - half_width) % 360.0
        pressurefix.loc[mask, "dir_95"] = (center + half_width) % 360.0
        changed += int(mask.sum())
        print(f"station policy {region} d{horizon}: rows={int(mask.sum()):,}; half_width={half_width}", flush=True)
    print(f"station long calibrated rows: {changed:,}", flush=True)
    write_stage(pressurefix, OUT_CSV, OUT_ZIP, "pressurefix_station_long_calibrated")


if __name__ == "__main__":
    main()
