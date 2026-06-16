from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
SAMPLE_PER_ANCHOR_DATE = int(os.environ.get("SEA_WINDS_ANCHOR_SAMPLE_PER_DATE", "350"))


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


def circular_mean_deg(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    ar = np.radians(a)
    br = np.radians(b)
    x = (1.0 - w_b) * np.cos(ar) + w_b * np.cos(br)
    y = (1.0 - w_b) * np.sin(ar) + w_b * np.sin(br)
    return np.degrees(np.arctan2(y, x)) % 360.0


def cws(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    good = np.isfinite(y) & np.isfinite(pred)
    y = y[good]
    pred = pred[good]
    best = SOL.optimize_dir_halfwidth(y, pred, SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def cache_tag(profile: str) -> str:
    return (
        "v6_speed_10m_100m_1000_925_850_700_500"
        "__dir_10m_100m_1000_925_850_700_500"
        "__cb_none"
        f"__profile_{profile}"
    )


def load_dir_bundle(region: str, profile: str = "quality_lgb_dirall") -> dict:
    p = CACHE / f"{region}_grid_dir_{cache_tag(profile)}.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def needed_feature_columns(bundle: dict) -> list[str]:
    cols = {"time", "latitude", "longitude"}
    for level_bundle in bundle["models"].values():
        for target_bundle in level_bundle.values():
            cols.update(target_bundle["features"])
    for h in (1, 7, 10):
        for hr in HOURS:
            cols.add(f"fcst_dir_d{h}_h{hr}")
            for lev in PRESSURE_LEVELS:
                cols.add(f"fcst_u_{lev}_d{h}_h{hr}")
                cols.add(f"fcst_v_{lev}_d{h}_h{hr}")
    return sorted(cols)


def forecast_dir_from_features(df: pd.DataFrame, level: str, horizon: int, hour: int) -> np.ndarray | None:
    hres_h = horizon if horizon in (1, 7) else 10
    if level in ("10m", "100m"):
        col = f"fcst_dir_d{hres_h}_h{hour}"
        if col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64") % 360.0
    u_col = f"fcst_u_{level}_d{hres_h}_h{hour}"
    v_col = f"fcst_v_{level}_d{hres_h}_h{hour}"
    if u_col not in df.columns or v_col not in df.columns:
        return None
    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype="float64")
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def load_surface100_lookup(region: str) -> pd.DataFrame:
    cols = ["time", "latitude", "longitude", "u100", "v100"]
    df = pd.read_parquet(DATA / "train" / f"reanalysis_{region}_6h.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["dir"] = (270.0 - np.degrees(np.arctan2(df["v100"], df["u100"]))) % 360.0
    return df[["time", "latitude", "longitude", "dir"]].set_index(["time", "latitude", "longitude"]).sort_index()


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
        df[f"dir_{lev}"] = (270.0 - np.degrees(np.arctan2(df[f"v_{lev}"], df[f"u_{lev}"]))) % 360.0
        keep.append(f"dir_{lev}")
    return df[keep].set_index(["time", "latitude", "longitude"]).sort_index()


def target_direction(
    eval_df: pd.DataFrame,
    level: str,
    horizon: int,
    hour: int,
    surf100_lookup: pd.DataFrame,
    pressure_lookup: pd.DataFrame,
) -> np.ndarray:
    if level == "10m":
        return pd.to_numeric(eval_df[f"dir_d{horizon}_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0

    future = eval_df["time"] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    keys = pd.MultiIndex.from_arrays(
        [future.values, eval_df["latitude"].values, eval_df["longitude"].values],
        names=["time", "latitude", "longitude"],
    )
    if level == "100m":
        return surf100_lookup["dir"].reindex(keys).to_numpy(dtype="float64") % 360.0
    return pressure_lookup[f"dir_{level}"].reindex(keys).to_numpy(dtype="float64") % 360.0


def predict_model_centers(eval_df: pd.DataFrame, bundle: dict) -> Dict[str, Dict[Tuple[int, int], np.ndarray]]:
    out: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for level in LEVELS:
        pred_df = SOL.predict_grid_direction_level(eval_df, bundle["models"][level], bundle["calibration"][level])
        out[level] = {}
        for h in HORIZONS:
            for hr in HOURS:
                sub = pred_df[(pred_df["horizon"] == h) & (pred_df["hour"] == hr)]
                out[level][(h, hr)] = sub["dir_50"].to_numpy(dtype="float64") % 360.0
    return out


def candidate_arrays(
    eval_df: pd.DataFrame,
    model_centers: Dict[str, Dict[Tuple[int, int], np.ndarray]],
    level: str,
    horizon: int,
    hour: int,
) -> Dict[str, np.ndarray]:
    base = model_centers[level][(horizon, hour)]
    cands = {"model_by_level": base, f"model_{level}": base}
    hres = forecast_dir_from_features(eval_df, level, horizon, hour)
    if hres is not None:
        cands["hres_by_level"] = hres
        cands[f"hres_{level}"] = hres
        for w in (0.25, 0.50, 0.75):
            cands[f"blend_model_hres_{w:.2f}"] = circular_mean_deg(base, hres, w)
    for proxy in ("500", "700", "850", "10m", "100m"):
        if proxy in model_centers:
            p = model_centers[proxy][(horizon, hour)]
            cands[f"model_{proxy}_proxy"] = p
            if proxy != level:
                for w in (0.25, 0.50, 0.75):
                    cands[f"blend_model_{proxy}_{w:.2f}"] = circular_mean_deg(base, p, w)
    return cands


def evaluate_region(region: str) -> pd.DataFrame:
    bundle = load_dir_bundle(region)
    cols = needed_feature_columns(bundle)
    # Include actual 10m target columns.
    for h in HORIZONS:
        for hr in HOURS:
            cols.append(f"dir_d{h}_h{hr}")
    cols = sorted(set(cols))
    train = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=cols)
    train["time"] = pd.to_datetime(train["time"])
    train["latitude"] = train["latitude"].astype(float).round(2)
    train["longitude"] = train["longitude"].astype(float).round(2)
    eval_df = train[train["time"].isin(ANCHOR_DATES_2021)].copy().reset_index(drop=True)
    if SAMPLE_PER_ANCHOR_DATE > 0:
        parts = []
        for _, part in eval_df.groupby("time", sort=True):
            if len(part) > SAMPLE_PER_ANCHOR_DATE:
                parts.append(part.sample(SAMPLE_PER_ANCHOR_DATE, random_state=2026))
            else:
                parts.append(part)
        eval_df = pd.concat(parts, ignore_index=True)
    print(f"{region}: anchor eval rows={len(eval_df):,}", flush=True)

    surf100 = load_surface100_lookup(region)
    pressure = load_pressure_lookup(region)
    centers = predict_model_centers(eval_df, bundle)

    rows = []
    for group_name, group_levels in [
        ("surface", ("10m", "100m")),
        ("pressure", PRESSURE_LEVELS),
    ]:
        for horizon in HORIZONS:
            y_parts_by_level_hour = {}
            cand_parts: Dict[str, list[np.ndarray]] = {}
            for level in group_levels:
                for hour in HOURS:
                    y = target_direction(eval_df, level, horizon, hour, surf100, pressure)
                    y_parts_by_level_hour[(level, hour)] = y
                    for name, arr in candidate_arrays(eval_df, centers, level, horizon, hour).items():
                        cand_parts.setdefault(name, []).append(arr)
            y_cat = np.concatenate([y_parts_by_level_hour[k] for k in y_parts_by_level_hour])
            for name, parts in cand_parts.items():
                # Candidate must be present for every level/hour in this group.
                if len(parts) != len(y_parts_by_level_hour):
                    continue
                pred = np.concatenate(parts)
                score, width = cws(y_cat, pred)
                rows.append(
                    {
                        "region": region,
                        "group": group_name,
                        "horizon": horizon,
                        "candidate": name,
                        "score": score,
                        "half_width": width,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for region in ("north_sea", "east_china_sea"):
        res = evaluate_region(region)
        all_rows.append(res)
        print("\nBest by block:", region, flush=True)
        best = res.sort_values("score").groupby(["group", "horizon"]).head(8)
        print(best.to_string(index=False), flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = WORK / "direction_anchor_backtest_candidates.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
