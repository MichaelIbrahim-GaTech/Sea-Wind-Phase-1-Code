from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

REGIONS = ("north_sea", "east_china_sea")
HOURS = (0, 6, 12, 18)
HORIZONS = (7, 14)
SURFACE_LEVELS = ("10m", "100m")
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
YEARS = (2020, 2021)
SAMPLE_PER_ANCHOR = 500


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
    ok = np.isfinite(y) & np.isfinite(pred)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    best = SOL.optimize_dir_halfwidth(y[ok], pred[ok], SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def surface_actual(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
    df = pd.read_parquet(DATA / "train" / f"reanalysis_{region}_6h.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["dir_10m"] = (270.0 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360.0
    df["dir_100m"] = (270.0 - np.degrees(np.arctan2(df["v100"], df["u100"]))) % 360.0
    return df[["time", "latitude", "longitude", "dir_10m", "dir_100m"]].set_index(["time", "latitude", "longitude"]).sort_index()


def pressure_actual(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        cols.extend([f"u_{level}", f"v_{level}"])
    df = pd.read_parquet(DATA / "train" / f"reanalysis_pressure_{region}.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    keep = ["time", "latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        df[f"dir_{level}"] = (270.0 - np.degrees(np.arctan2(df[f"v_{level}"], df[f"u_{level}"]))) % 360.0
        keep.append(f"dir_{level}")
    return df[keep].set_index(["time", "latitude", "longitude"]).sort_index()


def hres_dir(df: pd.DataFrame, level: str, horizon: int, hour: int) -> np.ndarray:
    lead = horizon if horizon == 7 else 10
    if level in SURFACE_LEVELS:
        return pd.to_numeric(df[f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0
    u = pd.to_numeric(df[f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df[f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def actual_for(
    anchor_rows: pd.DataFrame,
    lookup: pd.DataFrame,
    level: str,
    horizon: int,
    hour: int,
) -> np.ndarray:
    future = anchor_rows["time"] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    keys = pd.MultiIndex.from_arrays(
        [future.values, anchor_rows["latitude"].values, anchor_rows["longitude"].values],
        names=["time", "latitude", "longitude"],
    )
    return lookup[f"dir_{level}"].reindex(keys).to_numpy(dtype="float64") % 360.0


def evaluate_region_year(region: str, year: int) -> pd.DataFrame:
    needed = {"time", "latitude", "longitude"}
    for h in HORIZONS:
        lead = h if h == 7 else 10
        for hour in HOURS:
            needed.add(f"fcst_dir_d{lead}_h{hour}")
            for level in PRESSURE_LEVELS:
                needed.add(f"fcst_u_{level}_d{lead}_h{hour}")
                needed.add(f"fcst_v_{level}_d{lead}_h{hour}")

    train = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=sorted(needed))
    train["time"] = pd.to_datetime(train["time"])
    train["latitude"] = train["latitude"].astype(float).round(2)
    train["longitude"] = train["longitude"].astype(float).round(2)
    anchors = pd.to_datetime([f"{year}-{mmdd}" for mmdd in ANCHOR_MMDD])
    eval_df = train[train["time"].isin(anchors)].copy()
    parts = []
    for _, part in eval_df.groupby("time", sort=True):
        parts.append(part.sample(min(len(part), SAMPLE_PER_ANCHOR), random_state=year))
    eval_df = pd.concat(parts, ignore_index=True)

    surface = surface_actual(region)
    pressure = pressure_actual(region)
    rows = []
    for group, levels, lookup in [
        ("surface", SURFACE_LEVELS, surface),
        ("pressure", PRESSURE_LEVELS, pressure),
    ]:
        for horizon in HORIZONS:
            y_parts = []
            pred_parts = []
            for level in levels:
                for hour in HOURS:
                    y_parts.append(actual_for(eval_df, lookup, level, horizon, hour))
                    pred_parts.append(hres_dir(eval_df, level, horizon, hour))
            y = np.concatenate(y_parts)
            pred = np.concatenate(pred_parts)
            score, hw = cws(y, pred)
            rows.append(
                {
                    "region": region,
                    "year": year,
                    "group": group,
                    "horizon": horizon,
                    "candidate": "hres_by_level",
                    "score": score,
                    "half_width": hw,
                    "n": int(np.isfinite(y).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    for region in REGIONS:
        for year in YEARS:
            print(f"Evaluating {region} {year}", flush=True)
            rows.append(evaluate_region_year(region, year))
    out = pd.concat(rows, ignore_index=True)
    summary = (
        out.groupby(["region", "group", "horizon", "candidate"], as_index=False)
        .agg(score_mean=("score", "mean"), score_max=("score", "max"), half_width_mean=("half_width", "mean"))
        .sort_values(["region", "group", "horizon"])
    )
    out_path = WORK / "robust_hres_direction_backtest_by_year.csv"
    summary_path = WORK / "robust_hres_direction_backtest_summary.csv"
    out.to_csv(out_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(out.to_string(index=False), flush=True)
    print("\nSummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
