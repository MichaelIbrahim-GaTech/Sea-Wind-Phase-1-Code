from __future__ import annotations

import json
import zipfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path("runs/v6_pressure_speed")
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

DEFAULT_BASE_CSV = WORK / "predictions_broad_grid_dir_anchorblend_keepwidth_compact.csv"
MODEL_CSV = WORK / "predictions_direction_all_station_refine_compact.csv"
DEFAULT_OUT_CSV = WORK / "predictions_ns_grid_dir_p14s7_stationns_compact.csv"
DEFAULT_OUT_ZIP = WORK / "submission_ns_grid_dir_p14s7_stationns_compact.zip"

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
GRID_KEY = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]
PRESSURE_LEVELS = ["1000", "925", "850", "700", "500"]
HOURS = [0, 6, 12, 18]


def normalize(df: pd.DataFrame) -> None:
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    if "station" in df.columns:
        df["station"] = df["station"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)


def recenter_keep_width(df: pd.DataFrame, mask: pd.Series, center: np.ndarray) -> int:
    idx = df.index[mask]
    old_lo = pd.to_numeric(df.loc[idx, "dir_05"], errors="coerce").to_numpy(dtype="float64") % 360.0
    old_hi = pd.to_numeric(df.loc[idx, "dir_95"], errors="coerce").to_numpy(dtype="float64") % 360.0
    half_width = ((old_hi - old_lo) % 360.0) / 2.0
    center = np.asarray(center, dtype="float64") % 360.0
    df.loc[idx, "dir_50"] = center
    df.loc[idx, "dir_05"] = (center - half_width) % 360.0
    df.loc[idx, "dir_95"] = (center + half_width) % 360.0
    return len(idx)


def load_hres_surface_d7_ns() -> pd.DataFrame:
    rows = []
    for window in range(1, 9):
        cols = ["latitude", "longitude"] + [f"fcst_dir_d7_h{hr}" for hr in HOURS]
        feat = pd.read_parquet(FEATURES / f"inference_window_{window}_north_sea.parquet", columns=cols)
        feat["window"] = window
        feat["region"] = "north_sea"
        feat["latitude"] = feat["latitude"].astype(float).round(2)
        feat["longitude"] = feat["longitude"].astype(float).round(2)
        for hr in HOURS:
            tmp = feat[["window", "region", "latitude", "longitude", f"fcst_dir_d7_h{hr}"]].copy()
            tmp["horizon"] = 7
            tmp["hour"] = hr
            tmp["hres_surface_d7"] = pd.to_numeric(tmp[f"fcst_dir_d7_h{hr}"], errors="coerce") % 360.0
            rows.append(tmp[["window", "region", "latitude", "longitude", "horizon", "hour", "hres_surface_d7"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        ["window", "region", "latitude", "longitude", "horizon", "hour"]
    )


def load_model700_proxy() -> pd.DataFrame:
    model = pd.read_csv(MODEL_CSV, usecols=GRID_KEY + ["dir_50"], low_memory=False)
    normalize(model)
    model["dir_50"] = pd.to_numeric(model["dir_50"], errors="coerce") % 360.0
    model = model[model["type"].eq("grid") & model["level"].eq("700")].copy()
    proxy_key = ["type", "window", "region", "latitude", "longitude", "horizon", "hour"]
    return model[proxy_key + ["dir_50"]].rename(columns={"dir_50": "model700_center"})


def circ_mean_deg(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(np.radians(arr)).mean(), np.cos(np.radians(arr)).mean())) % 360.0)


def read_anchor(window: int) -> pd.Timestamp:
    meta_path = DATA / "inference" / f"window_{window}" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    return pd.Timestamp(meta["context_end"])


def load_station_history(window: int) -> pd.DataFrame:
    region = "north_sea"
    train = pd.read_parquet(DATA / "train" / f"stations_{region}_6h.parquet")
    ctx = pd.read_parquet(DATA / "inference" / f"window_{window}" / f"context_stations_{region}.parquet")
    hist = pd.concat([train, ctx], ignore_index=True)
    hist["time"] = pd.to_datetime(hist["time"])
    hist["hour"] = hist["time"].dt.hour.astype("int8")
    hist["station"] = hist["station"].astype(str)
    hist["direction"] = pd.to_numeric(hist["direction"], errors="coerce") % 360.0
    return hist


def recent14_same_hour(hist: pd.DataFrame, station: str, anchor: pd.Timestamp, hour: int) -> float:
    sub = hist[
        hist["station"].eq(station)
        & hist["hour"].eq(hour)
        & hist["time"].le(anchor + pd.Timedelta(hours=18))
        & hist["time"].ge(anchor - pd.Timedelta(days=13))
    ]
    return circ_mean_deg(sub["direction"])


def apply_station_ns_d14(df: pd.DataFrame) -> int:
    changed = 0
    half_width = 135.0
    for window in range(1, 9):
        anchor = read_anchor(window)
        hist = load_station_history(window)
        m = (
            df["type"].eq("station")
            & df["window"].eq(window)
            & df["region"].eq("north_sea")
            & df["horizon"].eq(14)
        )
        for row_idx in df.index[m]:
            center = recent14_same_hour(hist, str(df.at[row_idx, "station"]), anchor, int(df.at[row_idx, "hour"]))
            if not np.isfinite(center):
                continue
            df.at[row_idx, "dir_50"] = center % 360.0
            df.at[row_idx, "dir_05"] = (center - half_width) % 360.0
            df.at[row_idx, "dir_95"] = (center + half_width) % 360.0
            changed += 1
    return changed


def compact_and_validate(df: pd.DataFrame) -> None:
    for c in ["q05", "q50", "q95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).round(2)
    df["q05"] = df[["q05", "q50"]].min(axis=1).round(2)
    df["q95"] = df[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)

    grid = df["type"].eq("grid")
    counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[["q05", "q50", "q95"]] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]].isna().any(axis=1).sum())
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE_CSV), help="Input predictions CSV to postprocess")
    ap.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV), help="Output compact predictions CSV")
    ap.add_argument("--output-zip", default=str(DEFAULT_OUT_ZIP), help="Output zip containing root predictions.csv")
    args = ap.parse_args()
    base_csv = Path(args.base)
    out_csv = Path(args.output_csv)
    out_zip = Path(args.output_zip)

    print(f"Reading base {base_csv} ({base_csv.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(base_csv, low_memory=False)[COLS].copy()
    normalize(df)

    print("Loading North Sea HRES d7 surface direction maps", flush=True)
    hres_surface = load_hres_surface_d7_ns()
    df["hres_surface_d7"] = np.nan
    grid = df["type"].eq("grid")
    surf_lookup = df.loc[grid].reset_index()[
        ["index", "window", "region", "latitude", "longitude", "horizon", "hour"]
    ].merge(
        hres_surface,
        on=["window", "region", "latitude", "longitude", "horizon", "hour"],
        how="left",
        validate="many_to_one",
    )
    df.loc[surf_lookup["index"].to_numpy(), "hres_surface_d7"] = surf_lookup["hres_surface_d7"].to_numpy()

    print("Loading model 700-level proxy direction centers", flush=True)
    model700 = load_model700_proxy()
    df["model700_center"] = np.nan
    proxy_key = ["type", "window", "region", "latitude", "longitude", "horizon", "hour"]
    model_lookup = df.loc[grid].reset_index()[["index"] + proxy_key].merge(
        model700,
        on=proxy_key,
        how="left",
        validate="many_to_one",
    )
    df.loc[model_lookup["index"].to_numpy(), "model700_center"] = model_lookup["model700_center"].to_numpy()

    patch_counts = {}
    m = grid & df["region"].eq("north_sea") & df["level"].isin(["10m", "100m"]) & df["horizon"].eq(7)
    if df.loc[m, "hres_surface_d7"].isna().any():
        raise SystemExit("missing North Sea surface d7 HRES centers")
    patch_counts["ns_surface_d7_hres_center"] = recenter_keep_width(df, m, df.loc[m, "hres_surface_d7"].to_numpy(dtype="float64"))

    m = grid & df["region"].eq("north_sea") & df["level"].isin(PRESSURE_LEVELS) & df["horizon"].eq(14)
    if df.loc[m, "model700_center"].isna().any():
        raise SystemExit("missing North Sea pressure d14 model700 centers")
    patch_counts["ns_pressure_d14_model700_proxy"] = recenter_keep_width(df, m, df.loc[m, "model700_center"].to_numpy(dtype="float64"))

    patch_counts["ns_station_d14_recent14_direction"] = apply_station_ns_d14(df)
    print(f"Patch counts: {patch_counts}", flush=True)

    df = df.drop(columns=["hres_surface_d7", "model700_center"])
    compact_and_validate(df)

    print(f"Writing {out_csv}", flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(out_csv, index=False)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(out_csv, arcname="predictions.csv")
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    print(f"csv_size={out_csv.stat().st_size:,}; zip_size={out_zip.stat().st_size:,}; names={names}; uncompressed={info.file_size:,}", flush=True)
    print(f"OK: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
