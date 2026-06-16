from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "predictions_broad_robust_direction_v1_compact.csv"
STATION_ANEN_CSV = WORK / "predictions_station_vector_anen_v1_compact.csv"
OUT_CSV = WORK / "predictions_anen_hybrid_v1_compact.csv"
OUT_ZIP = WORK / "submission_anen_hybrid_v1_compact.zip"

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def normalize(df: pd.DataFrame) -> None:
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    df["station"] = df["station"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)


def same_order_or_die(base: pd.DataFrame, source: pd.DataFrame) -> None:
    keys = ["type", "window", "region", "latitude", "longitude", "station", "horizon", "hour", "level"]
    lhs = base[keys].fillna("__NA__").astype(str)
    rhs = source[keys].fillna("__NA__").astype(str)
    if not bool((lhs.values == rhs.values).all()):
        raise SystemExit("base and source are not in the same row order")


def validate(df: pd.DataFrame) -> None:
    for c in SPEED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).round(2)
    df["q05"] = df[["q05", "q50"]].min(axis=1).round(2)
    df["q95"] = df[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)
    grid = df["type"].eq("grid")
    counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    grid_dup = int(df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(df.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    print("Validation:", flush=True)
    print(f"  rows={len(df):,}; type_counts={counts}", flush=True)
    print(f"  bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    if len(df) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise SystemExit("row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise SystemExit("content validation failed")


def main() -> None:
    print(f"Reading broad base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[COLS].copy()
    print(f"Reading station AnEn source {STATION_ANEN_CSV} ({STATION_ANEN_CSV.stat().st_size:,} bytes)", flush=True)
    src = pd.read_csv(STATION_ANEN_CSV, low_memory=False)[COLS].copy()
    normalize(df)
    normalize(src)
    same_order_or_die(df, src)

    mask = (
        df["type"].eq("station")
        & df["region"].eq("east_china_sea")
        & df["horizon"].eq(1)
    )
    df.loc[mask, DIR_COLS] = src.loc[mask, DIR_COLS].to_numpy()
    print(f"Copied ECS station d1 vector-AnEn direction rows: {int(mask.sum()):,}", flush=True)
    validate(df)
    print(f"Writing {OUT_CSV}", flush=True)
    df[COLS].to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    print(f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; names={names}; uncompressed={info.file_size:,}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
