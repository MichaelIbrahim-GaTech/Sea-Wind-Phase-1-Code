from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"
CACHE = WORK / "model_cache"

HORIZONS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
LEVELS = ("10m", "100m", "1000", "925", "850", "700", "500")
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
ANCHOR_DATES_2021 = pd.to_datetime(
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
SAMPLE_PER_ANCHOR_DATE = int(os.environ.get("SEA_WINDS_SPEED_ANCHOR_SAMPLE_PER_DATE", "260"))


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


def cache_tag(profile: str) -> str:
    return (
        "v6_speed_10m_100m_1000_925_850_700_500"
        "__dir_10m_100m_1000_925_850_700_500"
        "__cb_none"
        f"__profile_{profile}"
    )


def load_speed_bundle(region: str, profile: str = "quality_lgb_dirall") -> dict:
    p = CACHE / f"{region}_grid_speed_{cache_tag(profile)}.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def needed_feature_columns(bundle: dict) -> list[str]:
    cols = {"time", "latitude", "longitude"}
    for level_bundle in bundle["models"].values():
        for target_bundle in level_bundle.values():
            cols.update(target_bundle["features"])
    for h in (1, 7, 10):
        for hr in HOURS:
            cols.add(f"fcst_speed_d{h}_h{hr}")
            for lev in PRESSURE_LEVELS:
                cols.add(f"fcst_u_{lev}_d{h}_h{hr}")
                cols.add(f"fcst_v_{lev}_d{h}_h{hr}")
    return sorted(cols)


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    good = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(SOL.winkler_score_np(y[good], lo[good], hi[good], alpha=0.10))


def optimize_symmetric(y: np.ndarray, center: np.ndarray, width_grid: np.ndarray) -> Tuple[float, float]:
    good = np.isfinite(y) & np.isfinite(center)
    y = y[good]
    center = center[good]
    best_score = float("inf")
    best_hw = float(width_grid[0])
    for hw in width_grid:
        lo = np.maximum(0.0, center - hw)
        hi = center + hw
        score = winkler(y, lo, hi)
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    return best_score, best_hw


def forecast_speed_from_features(df: pd.DataFrame, level: str, horizon: int, hour: int) -> np.ndarray | None:
    hres_h = horizon if horizon in (1, 7) else 10
    if level in ("10m", "100m"):
        col = f"fcst_speed_d{hres_h}_h{hour}"
        if col not in df.columns:
            return None
        base = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        if level == "100m":
            # HRES surface forecast in the starting kit is the 10m-like wind-speed field.
            base = base * 1.25
        return base
    u_col = f"fcst_u_{level}_d{hres_h}_h{hour}"
    v_col = f"fcst_v_{level}_d{hres_h}_h{hour}"
    if u_col not in df.columns or v_col not in df.columns:
        return None
    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype="float64")
    return np.sqrt(u * u + v * v)


def load_surface100_lookup(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude", "u100", "v100"]
    df = pd.read_parquet(DATA / "train" / f"reanalysis_{region}_6h.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["speed"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)
    return df[["time", "latitude", "longitude", "speed"]].set_index(["time", "latitude", "longitude"]).sort_index()


def load_pressure_lookup(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude"]
    for lev in PRESSURE_LEVELS:
        cols.extend([f"u_{lev}", f"v_{lev}"])
    df = pd.read_parquet(DATA / "train" / f"reanalysis_pressure_{region}.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    keep = ["time", "latitude", "longitude"]
    for lev in PRESSURE_LEVELS:
        df[f"speed_{lev}"] = np.sqrt(df[f"u_{lev}"] ** 2 + df[f"v_{lev}"] ** 2)
        keep.append(f"speed_{lev}")
    return df[keep].set_index(["time", "latitude", "longitude"]).sort_index()


def target_speed(
    eval_df: pd.DataFrame,
    level: str,
    horizon: int,
    hour: int,
    surf100_lookup: pd.DataFrame,
    pressure_lookup: pd.DataFrame,
) -> np.ndarray:
    if level == "10m":
        return pd.to_numeric(eval_df[f"speed_d{horizon}_h{hour}"], errors="coerce").to_numpy(dtype="float64")

    future = eval_df["time"] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    keys = pd.MultiIndex.from_arrays(
        [future.values, eval_df["latitude"].values, eval_df["longitude"].values],
        names=["time", "latitude", "longitude"],
    )
    if level == "100m":
        return surf100_lookup["speed"].reindex(keys).to_numpy(dtype="float64")
    return pressure_lookup[f"speed_{level}"].reindex(keys).to_numpy(dtype="float64")


def predict_model_quantiles(eval_df: pd.DataFrame, bundle: dict) -> Dict[str, Dict[Tuple[int, int], pd.DataFrame]]:
    out: Dict[str, Dict[Tuple[int, int], pd.DataFrame]] = {}
    for level in LEVELS:
        pred_df = SOL.predict_grid_speed_level(eval_df, bundle["models"][level], bundle["calibration"][level])
        out[level] = {}
        for h in HORIZONS:
            for hr in HOURS:
                sub = pred_df[(pred_df["horizon"] == h) & (pred_df["hour"] == hr)][["q05", "q50", "q95"]].reset_index(drop=True)
                out[level][(h, hr)] = sub
    return out


def evaluate_region(region: str) -> pd.DataFrame:
    bundle = load_speed_bundle(region)
    cols = needed_feature_columns(bundle)
    for h in HORIZONS:
        for hr in HOURS:
            cols.append(f"speed_d{h}_h{hr}")
    cols = sorted(set(cols))
    train = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=cols)
    train["time"] = pd.to_datetime(train["time"])
    train["latitude"] = train["latitude"].astype(float).round(2)
    train["longitude"] = train["longitude"].astype(float).round(2)
    eval_df = train[train["time"].isin(ANCHOR_DATES_2021)].copy().reset_index(drop=True)
    if SAMPLE_PER_ANCHOR_DATE > 0:
        parts = []
        for _, part in eval_df.groupby("time", sort=True):
            parts.append(part.sample(min(len(part), SAMPLE_PER_ANCHOR_DATE), random_state=2031))
        eval_df = pd.concat(parts, ignore_index=True)
    print(f"{region}: anchor eval rows={len(eval_df):,}", flush=True)

    surf100 = load_surface100_lookup(region)
    pressure = load_pressure_lookup(region)
    model_q = predict_model_quantiles(eval_df, bundle)
    width_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0])

    rows = []
    for group_name, group_levels in [
        ("surface", ("10m", "100m")),
        ("pressure", PRESSURE_LEVELS),
    ]:
        for horizon in HORIZONS:
            y_parts = []
            model_lo_parts = []
            model_mid_parts = []
            model_hi_parts = []
            hres_parts = []
            for level in group_levels:
                for hour in HOURS:
                    y_parts.append(target_speed(eval_df, level, horizon, hour, surf100, pressure))
                    q = model_q[level][(horizon, hour)]
                    model_lo_parts.append(q["q05"].to_numpy(dtype="float64"))
                    model_mid_parts.append(q["q50"].to_numpy(dtype="float64"))
                    model_hi_parts.append(q["q95"].to_numpy(dtype="float64"))
                    hres_parts.append(forecast_speed_from_features(eval_df, level, horizon, hour))

            y = np.concatenate(y_parts)
            model_lo = np.concatenate(model_lo_parts)
            model_mid = np.concatenate(model_mid_parts)
            model_hi = np.concatenate(model_hi_parts)
            hres = np.concatenate(hres_parts)
            rows.append(
                {
                    "region": region,
                    "group": group_name,
                    "horizon": horizon,
                    "candidate": "current_model_interval",
                    "score": winkler(y, model_lo, model_hi),
                    "half_width": float(np.nanmedian((model_hi - model_lo) / 2.0)),
                }
            )
            for name, center in [("hres_center", hres)]:
                score, hw = optimize_symmetric(y, center, width_grid)
                rows.append({"region": region, "group": group_name, "horizon": horizon, "candidate": name, "score": score, "half_width": hw})
            for w in (0.25, 0.50, 0.75):
                center = (1.0 - w) * model_mid + w * hres
                score, hw = optimize_symmetric(y, center, width_grid)
                rows.append(
                    {
                        "region": region,
                        "group": group_name,
                        "horizon": horizon,
                        "candidate": f"blend_model_hres_{w:.2f}",
                        "score": score,
                        "half_width": hw,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for region in ("north_sea", "east_china_sea"):
        res = evaluate_region(region)
        all_rows.append(res)
        print(f"\nBest speed candidates for {region}", flush=True)
        print(res.sort_values("score").groupby(["group", "horizon"]).head(5).to_string(index=False), flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    out_path = WORK / "speed_anchor_backtest_candidates.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
