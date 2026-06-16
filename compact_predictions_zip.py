from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input predictions.csv")
    ap.add_argument("--output-csv", required=True, help="Output compact CSV")
    ap.add_argument("--output-zip", required=True, help="Output zip containing predictions.csv")
    ap.add_argument("--speed-dp", type=int, default=2)
    ap.add_argument("--dir-dp", type=int, default=1)
    args = ap.parse_args()

    inp = Path(args.input)
    out_csv = Path(args.output_csv)
    out_zip = Path(args.output_zip)
    print(f"Reading {inp} ({inp.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(inp, low_memory=False)[COLS].copy()
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["station"] = df["station"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    for c in ["q05", "q50", "q95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).round(args.speed_dp)
    df["q05"] = df[["q05", "q50"]].min(axis=1).round(args.speed_dp)
    df["q95"] = df[["q95", "q50"]].max(axis=1).round(args.speed_dp)
    for c in DIR_COLS:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") % 360.0).round(args.dir_dp) % 360.0).round(args.dir_dp)

    grid = df["type"].eq("grid")
    counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[["q05", "q50", "q95"]] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]].isna().any(axis=1).sum())
    grid_dup = int(df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(df.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    print(f"rows={len(df):,}; type_counts={counts}", flush=True)
    print(f"bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    if len(df) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise SystemExit("row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise SystemExit("content validation failed")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_csv}", flush=True)
    df.to_csv(out_csv, index=False)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(out_csv, arcname="predictions.csv")
    with zipfile.ZipFile(out_zip) as zf:
        info = zf.getinfo("predictions.csv")
    print(f"csv_size={out_csv.stat().st_size:,}; zip_size={out_zip.stat().st_size:,}; uncompressed={info.file_size:,}", flush=True)
    print(f"OK: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
