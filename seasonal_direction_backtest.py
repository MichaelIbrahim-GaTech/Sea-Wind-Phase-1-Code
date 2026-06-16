from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

HOURS = (0, 6, 12, 18)
HORIZONS = (7, 14)
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
ANCHORS_2021 = pd.to_datetime(
    [
        "2021-01-14",
        "2021-02-25",
        "2021-04-08",
        "2021-05-20",
        "2021-07-01",
        "2021-08-12",
        "2021-09-23",
        "2021-11-04",
    ]
)
SAMPLE_PER_ANCHOR_DATE = int(os.environ.get("SEA_WINDS_SEASONAL_SAMPLE_PER_DATE", "400"))
WINDOWS = tuple(int(x) for x in os.environ.get("SEA_WINDS_SEASONAL_WINDOWS", "7,14,21,30,45,60").split(","))


def load_solution_module():
    path = ROOT / "sea_winds_solution_ephemeral_v6_pressure_speed.py"
    spec = importlib.util.spec_from_file_location("sea_winds_solution_v6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SOL = load_solution_module()


def cws(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(y) & np.isfinite(pred)
    best = SOL.optimize_dir_halfwidth(y[good], pred[good], SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def doy_distance(a: pd.Series, b: int) -> pd.Series:
    d = (a.astype(int) - int(b)).abs()
    return np.minimum(d, 366 - d)


def prepare_surface_actual(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
    df = pd.read_parquet(DATA / "train" / f"reanalysis_{region}_6h.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["year"] = df["time"].dt.year.astype("int16")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["dir_10m"] = (270.0 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360.0
    df["dir_100m"] = (270.0 - np.degrees(np.arctan2(df["v100"], df["u100"]))) % 360.0
    return df[["time", "latitude", "longitude", "year", "doy", "hour", "dir_10m", "dir_100m"]]


def prepare_pressure_actual(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        cols.extend([f"u_{level}", f"v_{level}"])
    df = pd.read_parquet(DATA / "train" / f"reanalysis_pressure_{region}.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["year"] = df["time"].dt.year.astype("int16")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    df["hour"] = df["time"].dt.hour.astype("int8")
    keep = ["time", "latitude", "longitude", "year", "doy", "hour"]
    for level in PRESSURE_LEVELS:
        df[f"dir_{level}"] = (270.0 - np.degrees(np.arctan2(df[f"v_{level}"], df[f"u_{level}"]))) % 360.0
        keep.append(f"dir_{level}")
    return df[keep]


def seasonal_center(
    actual: pd.DataFrame,
    level_col: str,
    target_time,
    eval_coords: pd.DataFrame,
    years: Iterable[int],
    window_days: int,
) -> np.ndarray:
    target_time = pd.Timestamp(target_time)
    subset = actual[
        actual["year"].isin(list(years))
        & actual["hour"].eq(target_time.hour)
        & (doy_distance(actual["doy"], target_time.dayofyear) <= int(window_days))
    ]
    if subset.empty:
        return np.full(len(eval_coords), np.nan)
    ang = np.radians(pd.to_numeric(subset[level_col], errors="coerce").to_numpy(dtype="float64"))
    tmp = subset[["latitude", "longitude"]].copy()
    tmp["sin"] = np.sin(ang)
    tmp["cos"] = np.cos(ang)
    grp = tmp.groupby(["latitude", "longitude"], sort=False)[["sin", "cos"]].mean().reset_index()
    grp["seasonal_dir"] = np.degrees(np.arctan2(grp["sin"], grp["cos"])) % 360.0
    merged = eval_coords.merge(grp[["latitude", "longitude", "seasonal_dir"]], on=["latitude", "longitude"], how="left")
    return merged["seasonal_dir"].to_numpy(dtype="float64")


def seasonal_centers_for_windows(
    actual_hist_by_hour: dict[int, pd.DataFrame],
    level_col: str,
    target_time,
    eval_coords: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Circular seasonal means for all configured windows, scanning history once."""
    target_time = pd.Timestamp(target_time)
    subset = actual_hist_by_hour.get(int(target_time.hour))
    if subset is None or subset.empty:
        return {w: np.full(len(eval_coords), np.nan) for w in WINDOWS}

    dist = doy_distance(subset["doy"], target_time.dayofyear)
    keep = dist <= max(WINDOWS)
    if not bool(keep.any()):
        return {w: np.full(len(eval_coords), np.nan) for w in WINDOWS}

    tmp = subset.loc[keep, ["latitude", "longitude", level_col]].copy()
    tmp["doy_dist"] = dist.loc[keep].to_numpy(dtype="int16")
    ang = np.radians(pd.to_numeric(tmp[level_col], errors="coerce").to_numpy(dtype="float64"))
    tmp["sin"] = np.sin(ang)
    tmp["cos"] = np.cos(ang)

    out = {}
    for window_days in WINDOWS:
        win = tmp[tmp["doy_dist"].le(int(window_days))]
        if win.empty:
            out[window_days] = np.full(len(eval_coords), np.nan)
            continue
        grp = win.groupby(["latitude", "longitude"], sort=False)[["sin", "cos"]].mean().reset_index()
        grp["seasonal_dir"] = np.degrees(np.arctan2(grp["sin"], grp["cos"])) % 360.0
        merged = eval_coords.merge(grp[["latitude", "longitude", "seasonal_dir"]], on=["latitude", "longitude"], how="left")
        out[window_days] = merged["seasonal_dir"].to_numpy(dtype="float64")
    return out


def target_from_actual(actual: pd.DataFrame, level_col: str, target_time, eval_coords: pd.DataFrame) -> np.ndarray:
    target_time = pd.Timestamp(target_time)
    sub = actual[actual["time"].eq(target_time)][["latitude", "longitude", level_col]]
    merged = eval_coords.merge(sub, on=["latitude", "longitude"], how="left")
    return pd.to_numeric(merged[level_col], errors="coerce").to_numpy(dtype="float64") % 360.0


def load_model_predictions(base_csv: Path) -> pd.DataFrame:
    cols = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level", "dir_50"]
    if not base_csv.exists():
        raise SystemExit(f"missing base CSV: {base_csv}")
    df = pd.read_csv(base_csv, usecols=cols, low_memory=False)
    df = df[df["type"].astype(str).str.lower().eq("grid")].copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(2)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(2)
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("int64")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("int64")
    df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("int64")
    df["level"] = df["level"].astype(str)
    df["dir_50"] = pd.to_numeric(df["dir_50"], errors="coerce") % 360.0
    return df


def evaluate_region(region: str, model_pred: pd.DataFrame) -> pd.DataFrame:
    print(f"\nPreparing actual direction tables for {region}", flush=True)
    surface = prepare_surface_actual(region)
    pressure = prepare_pressure_actual(region)

    feat_cols = ["time", "latitude", "longitude"]
    feat = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=feat_cols)
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = feat["latitude"].astype(float).round(2)
    feat["longitude"] = feat["longitude"].astype(float).round(2)
    eval_df = feat[feat["time"].isin(ANCHORS_2021)].copy()
    if SAMPLE_PER_ANCHOR_DATE > 0:
        eval_df = pd.concat(
            [
                p.sample(min(len(p), SAMPLE_PER_ANCHOR_DATE), random_state=2027)
                for _, p in eval_df.groupby("time", sort=True)
            ],
            ignore_index=True,
        )
    print(f"{region}: sampled anchor rows={len(eval_df):,}", flush=True)

    # Assign public-style window ids for model prediction lookup.
    window_map = {d: i + 1 for i, d in enumerate(ANCHORS_2021)}
    eval_df["window"] = eval_df["time"].map(window_map).astype("int64")
    coord_cols = ["latitude", "longitude"]

    rows = []
    for group, levels, actual_table in [
        ("surface", ("10m", "100m"), surface),
        ("pressure", PRESSURE_LEVELS, pressure),
    ]:
        actual_hist = actual_table[actual_table["year"].isin([2019, 2020])].copy()
        actual_hist_by_hour = {
            int(hour): part.reset_index(drop=True)
            for hour, part in actual_hist.groupby("hour", sort=False)
        }
        for horizon in HORIZONS:
            # Accumulate y/model/seasonal predictions across levels/hours/windows.
            y_parts = []
            model_parts = []
            seasonal_parts = {w: [] for w in WINDOWS}
            for anchor in ANCHORS_2021:
                anchor_rows = eval_df[eval_df["time"].eq(anchor)].copy()
                window_id = int(window_map[anchor])
                for hour in HOURS:
                    target_time = anchor + pd.Timedelta(days=horizon) + pd.Timedelta(hours=hour)
                    for level in levels:
                        level_col = f"dir_{level}"
                        y = target_from_actual(actual_table, level_col, target_time, anchor_rows[coord_cols])
                        y_parts.append(y)
                        mp = model_pred[
                            model_pred["region"].eq(region)
                            & model_pred["window"].eq(window_id)
                            & model_pred["horizon"].eq(horizon)
                            & model_pred["hour"].eq(hour)
                            & model_pred["level"].eq(level)
                        ][["latitude", "longitude", "dir_50"]]
                        merged_model = anchor_rows[coord_cols].merge(mp, on=["latitude", "longitude"], how="left")
                        model_parts.append(merged_model["dir_50"].to_numpy(dtype="float64"))
                        seasonal_by_window = seasonal_centers_for_windows(
                            actual_hist_by_hour,
                            level_col,
                            target_time,
                            anchor_rows[coord_cols],
                        )
                        for w in WINDOWS:
                            seasonal_parts[w].append(seasonal_by_window[w])
            y_cat = np.concatenate(y_parts)
            model_cat = np.concatenate(model_parts)
            score, width = cws(y_cat, model_cat)
            rows.append({"region": region, "group": group, "horizon": horizon, "candidate": "current_model", "score": score, "half_width": width})
            for w, parts in seasonal_parts.items():
                pred = np.concatenate(parts)
                score, width = cws(y_cat, pred)
                rows.append({"region": region, "group": group, "horizon": horizon, "candidate": f"seasonal_w{w}", "score": score, "half_width": width})
                for blend_w in (0.25, 0.50, 0.75):
                    pred_blend = SOL.blend_direction_deg(pred, model_cat, blend_w)
                    score_b, width_b = cws(y_cat, pred_blend)
                    rows.append(
                        {
                            "region": region,
                            "group": group,
                            "horizon": horizon,
                            "candidate": f"blend_seasonal_w{w}_{blend_w:.2f}",
                            "score": score_b,
                            "half_width": width_b,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=Path, default=WORK / "pred_sfc14_selw.csv")
    ap.add_argument("--out-csv", type=Path, default=WORK / "seasonal_direction_backtest_candidates.csv")
    args = ap.parse_args()

    model_pred = load_model_predictions(args.base_csv)
    all_rows = []
    for region in ("north_sea", "east_china_sea"):
        res = evaluate_region(region, model_pred)
        all_rows.append(res)
        print(f"\nBest seasonal candidates for {region}", flush=True)
        print(res.sort_values("score").groupby(["group", "horizon"]).head(8).to_string(index=False), flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    out_path = args.out_csv
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
