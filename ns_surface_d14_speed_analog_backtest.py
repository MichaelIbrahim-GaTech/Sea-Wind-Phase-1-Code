from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
TRAIN = DATA / "train"

LEVELS = ("10m", "100m")
HOURS = (0, 6, 12, 18)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")


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


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(SOL.winkler_score_np(y, lo, hi, alpha=0.10))


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


def candidate_times(target: pd.Timestamp, train_years: tuple[int, ...], half_window: int) -> list[pd.Timestamp]:
    out = []
    for year in train_years:
        center = pd.Timestamp(year=year, month=target.month, day=target.day, hour=target.hour)
        for offset in range(-half_window, half_window + 1):
            out.append(center + pd.Timedelta(days=offset))
    return out


def analog_quantiles(
    speed: dict[str, np.ndarray],
    time_to_idx: dict[pd.Timestamp, int],
    target: pd.Timestamp,
    train_years: tuple[int, ...],
    half_window: int,
    level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = [time_to_idx[t] for t in candidate_times(target, train_years, half_window) if t in time_to_idx]
    if len(idx) < 3:
        raise RuntimeError(f"not enough analog times for {target} train_years={train_years}")
    vals = speed[level][idx, :]
    return (
        np.nanquantile(vals, 0.05, axis=0),
        np.nanquantile(vals, 0.50, axis=0),
        np.nanquantile(vals, 0.95, axis=0),
    )


def actual(speed: dict[str, np.ndarray], time_to_idx: dict[pd.Timestamp, int], target: pd.Timestamp, level: str) -> np.ndarray:
    return speed[level][time_to_idx[target], :]


def evaluate() -> pd.DataFrame:
    latlon, time_to_idx, speed = load_surface_cube()
    rows = []
    for val_year, train_years in [(2020, (2019,)), (2021, (2019, 2020))]:
        for half_window in (3, 7, 14, 21, 30, 45):
            y_parts = []
            q05_parts = []
            q50_parts = []
            q95_parts = []
            for mmdd in ANCHOR_MMDD:
                origin = pd.Timestamp(f"{val_year}-{mmdd}")
                for hour in HOURS:
                    target = origin + pd.Timedelta(days=14, hours=hour)
                    for level in LEVELS:
                        q05, q50, q95 = analog_quantiles(speed, time_to_idx, target, train_years, half_window, level)
                        y_parts.append(actual(speed, time_to_idx, target, level))
                        q05_parts.append(q05)
                        q50_parts.append(q50)
                        q95_parts.append(q95)
            y = np.concatenate(y_parts)
            q05 = np.concatenate(q05_parts)
            q95 = np.concatenate(q95_parts)
            rows.append(
                {
                    "val_year": val_year,
                    "train_years": ",".join(str(y) for y in train_years),
                    "half_window_days": half_window,
                    "score": winkler(y, q05, q95),
                    "rows": int(len(y)),
                }
            )
    out = pd.DataFrame(rows)
    summary = (
        out.groupby("half_window_days", as_index=False)
        .agg(score_mean=("score", "mean"), score_max=("score", "max"), score_min=("score", "min"))
        .sort_values("score_mean")
    )
    out_path = WORK / "ns_surface_d14_speed_analog_backtest_by_year.csv"
    summary_path = WORK / "ns_surface_d14_speed_analog_backtest_summary.csv"
    out.to_csv(out_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(out.to_string(index=False), flush=True)
    print("\nSummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return summary


if __name__ == "__main__":
    evaluate()
