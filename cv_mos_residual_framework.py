#!/usr/bin/env python3
"""
Cross-validation harness for Sea Winds MOS/residual candidates.

This is a development tool, not a public-leaderboard patcher.

Compliance:
- Uses only official files under runs/v6_pressure_speed/phase1_dataset.
- Uses historical train rows for fitting and later historical anchor windows for
  validation.
- Does not read external datasets or evaluation targets.

The first implemented candidate family is grid HRES-MOS:
- speed: LightGBM quantile models for HRES speed residuals
- direction: LightGBM residual u/v models around provided HRES vectors

It evaluates candidates against raw provided-HRES anchors and, where available,
the existing validation backtest tables for the current model family. The output
is meant to decide which blocks deserve a final inference/submission branch.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("runs/v6_pressure_speed")
DATA = ROOT / "phase1_dataset"
FEATURES = DATA / "features"
TRAIN = DATA / "train"
OUT_DIR = ROOT / "cv_mos_residual"
OUT_SUMMARY = ROOT / "cv_mos_residual_framework_summary.csv"
OUT_BY_FOLD = ROOT / "cv_mos_residual_framework_by_fold.csv"
DEFAULT_BLOCKS = "north_sea:surface:7,north_sea:pressure:7,east_china_sea:pressure:7,east_china_sea:surface:14"

REGIONS = ("north_sea", "east_china_sea")
GROUP_LEVELS = {
    "surface": ("10m", "100m"),
    "pressure": ("1000", "925", "850", "700", "500"),
}
HORIZONS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")

BASE_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "t2m",
    "msl",
    "sshf",
    "z700",
    "z850",
    "sst",
    "blh",
    "cape",
    "ws10",
    "ws100",
    "wind_shear",
    "woy_sin",
    "woy_cos",
    "ws10_h6",
    "msl_h6",
    "t2m_h6",
    "ws10_h18",
    "msl_h18",
    "t2m_h18",
    "ws10_daily_max",
    "ws10_daily_mean",
    "ws10_daily_range",
    "ws10_lag1d",
    "ws10_lag3d",
    "ws10_lag7d",
    "msl_lag1d",
    "msl_lag3d",
    "msl_lag7d",
    "t2m_lag1d",
    "t2m_lag3d",
    "t2m_lag7d",
    "sshf_lag1d",
    "sshf_lag3d",
    "sshf_lag7d",
    "z700_lag1d",
    "z700_lag3d",
    "z700_lag7d",
    "ws10_rmean3d",
    "ws10_rstd3d",
    "ws10_rmean7d",
    "ws10_rstd7d",
    "elevation",
    "nao_proxy",
    "siberian_high",
    "icelandic_low",
    "ns_pressure_gradient",
    "ecs_pressure_gradient",
]

BASE_DIRECTION_FEATURES = [
    "wd10",
    "wd100",
    "wd10_h6",
    "wd10_h18",
    "wd10_lag1d",
    "wd10_lag3d",
    "wd10_lag7d",
]

DIR_HALF_WIDTH_GRID = np.array(list(range(15, 181, 5)) + [179.9], dtype="float64")
SPEED_SCALE_GRID = np.array([0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.20, 1.40, 1.65, 2.0], dtype="float64")


def log(msg: str) -> None:
    print(msg, flush=True)


def schema_names(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def hres_lead(horizon: int) -> int:
    return horizon if horizon in (1, 7) else 10


def uv_from_speed_dir(speed: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(np.asarray(direction, dtype="float64") % 360.0)
    speed = np.asarray(speed, dtype="float64")
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u.astype("float32"), v.astype("float32")


def speed_dir_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u64 = np.asarray(u, dtype="float64")
    v64 = np.asarray(v, dtype="float64")
    speed = np.sqrt(u64 * u64 + v64 * v64)
    direction = (270.0 - np.degrees(np.arctan2(v64, u64))) % 360.0
    return speed.astype("float32"), direction.astype("float32")


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    lo = np.asarray(lo, dtype="float64")
    hi = np.asarray(hi, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not bool(ok.any()):
        return float("nan")
    y = y[ok]
    lo = lo[ok]
    hi = hi[ok]
    below = y < lo
    above = y > hi
    score = (hi - lo) + 20.0 * (lo - y) * below + 20.0 * (y - hi) * above
    return float(np.mean(score))


def best_symmetric_speed_width(y: np.ndarray, center: np.ndarray, width_grid: np.ndarray = SPEED_SCALE_GRID) -> tuple[float, float]:
    y = np.asarray(y, dtype="float64")
    center = np.asarray(center, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(center)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    err = np.abs(y[ok] - center[ok])
    best_score = float("inf")
    best_hw = float("nan")
    for hw in width_grid:
        lo = np.maximum(0.0, center[ok] - hw)
        hi = center[ok] + hw
        score = winkler_score(y[ok], lo, hi)
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    # Add empirical quantile option, often better than the coarse grid.
    for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.93):
        hw = float(np.nanquantile(err, q))
        lo = np.maximum(0.0, center[ok] - hw)
        hi = center[ok] + hw
        score = winkler_score(y[ok], lo, hi)
        if score < best_score:
            best_score = score
            best_hw = hw
    return best_score, best_hw


def circular_winkler_score(y: np.ndarray, center: np.ndarray, half_width: float) -> float:
    y = np.asarray(y, dtype="float64") % 360.0
    center = np.asarray(center, dtype="float64") % 360.0
    ok = np.isfinite(y) & np.isfinite(center)
    if not bool(ok.any()):
        return float("nan")
    y = y[ok]
    center = center[ok]
    lo = (center - half_width) % 360.0
    hi = (center + half_width) % 360.0
    width = (hi - lo) % 360.0
    inside = ((y - lo) % 360.0) <= width
    miss = np.minimum(circ_abs_diff(y, lo), circ_abs_diff(y, hi))
    return float(np.mean(width + 20.0 * miss * (~inside)))


def best_direction_width(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    best_score = float("inf")
    best_hw = 90.0
    for hw in DIR_HALF_WIDTH_GRID:
        score = circular_winkler_score(y, center, float(hw))
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    return best_score, best_hw


@dataclass
class CubeStore:
    n_grid: int
    latlon: pd.DataFrame
    time_to_idx: dict[pd.Timestamp, int]
    u: dict[str, np.ndarray]
    v: dict[str, np.ndarray]


def load_cube(region: str, group: str) -> CubeStore:
    if group == "surface":
        path = TRAIN / f"reanalysis_{region}_6h.parquet"
        cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
        level_cols = {"10m": ("u10", "v10"), "100m": ("u100", "v100")}
    else:
        path = TRAIN / f"reanalysis_pressure_{region}.parquet"
        cols = ["time", "latitude", "longitude"]
        for level in GROUP_LEVELS["pressure"]:
            cols.extend([f"u_{level}", f"v_{level}"])
        level_cols = {level: (f"u_{level}", f"v_{level}") for level in GROUP_LEVELS["pressure"]}

    df = pd.read_parquet(path, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    df = df.sort_values(["time", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
    times = pd.Series(df["time"].unique()).sort_values().to_numpy()
    n_times = len(times)
    n_grid = int(len(df) // n_times)
    latlon = df.loc[: n_grid - 1, ["latitude", "longitude"]].reset_index(drop=True)
    u_arrays: dict[str, np.ndarray] = {}
    v_arrays: dict[str, np.ndarray] = {}
    for level, (u_col, v_col) in level_cols.items():
        u_arrays[level] = df[u_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
        v_arrays[level] = df[v_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}
    return CubeStore(n_grid=n_grid, latlon=latlon, time_to_idx=time_to_idx, u=u_arrays, v=v_arrays)


def hres_columns(group: str, levels: Iterable[str], horizons: Iterable[int]) -> list[str]:
    cols: list[str] = []
    for horizon in horizons:
        lead = hres_lead(horizon)
        for hour in HOURS:
            if group == "surface":
                cols += [f"fcst_speed_d{lead}_h{hour}", f"fcst_dir_d{lead}_h{hour}"]
            else:
                for level in levels:
                    cols += [f"fcst_u_{level}_d{lead}_h{hour}", f"fcst_v_{level}_d{lead}_h{hour}"]
    return sorted(set(cols))


def load_feature_df(region: str, extra_hres_cols: Iterable[str]) -> pd.DataFrame:
    path = FEATURES / f"train_{region}.parquet"
    available = set(schema_names(path))
    cols = ["time", "latitude", "longitude"]
    cols += [c for c in BASE_NUMERIC_FEATURES if c in available and c not in cols]
    cols += [c for c in BASE_DIRECTION_FEATURES if c in available and c not in cols]
    cols += [c for c in extra_hres_cols if c in available and c not in cols]
    df = pd.read_parquet(path, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    return df.reset_index(drop=True)


def attach_grid_index(df: pd.DataFrame, cube: CubeStore, region: str, group: str) -> pd.DataFrame:
    grid = cube.latlon.reset_index().rename(columns={"index": "grid_idx"})
    out = df.merge(grid, on=["latitude", "longitude"], how="left", sort=False)
    if out["grid_idx"].isna().any():
        bad = out.loc[out["grid_idx"].isna(), ["latitude", "longitude"]].drop_duplicates().head()
        raise RuntimeError(f"{region}/{group} rows missing target grid index:\n{bad}")
    out["grid_idx"] = out["grid_idx"].astype("int32")
    return out


def base_feature_matrix(df: pd.DataFrame, row_idx: np.ndarray) -> pd.DataFrame:
    src = df.iloc[row_idx]
    out = pd.DataFrame(index=np.arange(len(row_idx)))
    for c in BASE_NUMERIC_FEATURES:
        if c in src.columns:
            out[c] = pd.to_numeric(src[c], errors="coerce").to_numpy(dtype="float32")
    for c in BASE_DIRECTION_FEATURES:
        if c in src.columns:
            a = pd.to_numeric(src[c], errors="coerce").to_numpy(dtype="float64")
            rad = np.deg2rad(a % 360.0)
            out[f"{c}_sin"] = np.sin(rad).astype("float32")
            out[f"{c}_cos"] = np.cos(rad).astype("float32")
    return out


def hres_uv_from_rows(df: pd.DataFrame, row_idx: np.ndarray, group: str, level: str, horizon: int, hour: int) -> tuple[np.ndarray, np.ndarray]:
    lead = hres_lead(horizon)
    src = df.iloc[row_idx]
    if group == "surface":
        speed = pd.to_numeric(src[f"fcst_speed_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
        direction = pd.to_numeric(src[f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
        return uv_from_speed_dir(speed, direction)
    u = pd.to_numeric(src[f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
    v = pd.to_numeric(src[f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
    return u, v


def target_uv(cube: CubeStore, origin_times: np.ndarray, grid_idx: np.ndarray, level: str, horizon: int, hour: int) -> tuple[np.ndarray, np.ndarray]:
    future = pd.to_datetime(origin_times) + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    t_idx = np.array([cube.time_to_idx.get(pd.Timestamp(t), -1) for t in future], dtype="int32")
    ok = t_idx >= 0
    u = np.full(len(grid_idx), np.nan, dtype="float32")
    v = np.full(len(grid_idx), np.nan, dtype="float32")
    u[ok] = cube.u[level][t_idx[ok], grid_idx[ok]]
    v[ok] = cube.v[level][t_idx[ok], grid_idx[ok]]
    return u, v


def make_combo_matrix(
    df: pd.DataFrame,
    cube: CubeStore,
    origin_grid_rows: np.ndarray,
    region: str,
    group: str,
    horizon: int,
    levels: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_parts = []
    hres_u_parts = []
    hres_v_parts = []
    y_u_parts = []
    y_v_parts = []
    level_code = {level: i for i, level in enumerate(levels)}
    for level in levels:
        for hour in HOURS:
            row_idx = origin_grid_rows
            hu, hv = hres_uv_from_rows(df, row_idx, group, level, horizon, hour)
            yu, yv = target_uv(
                cube,
                df.iloc[row_idx]["time"].to_numpy(),
                df.iloc[row_idx]["grid_idx"].to_numpy(dtype="int32"),
                level,
                horizon,
                hour,
            )
            X = base_feature_matrix(df, row_idx)
            hspd, hdir = speed_dir_from_uv(hu, hv)
            X["hres_u"] = hu
            X["hres_v"] = hv
            X["hres_speed"] = hspd
            rad = np.deg2rad(hdir.astype("float64"))
            X["hres_dir_sin"] = np.sin(rad).astype("float32")
            X["hres_dir_cos"] = np.cos(rad).astype("float32")
            X["level_code"] = np.float32(level_code[level])
            X["hour_sin"] = np.float32(math.sin(2.0 * math.pi * hour / 24.0))
            X["hour_cos"] = np.float32(math.cos(2.0 * math.pi * hour / 24.0))
            X["horizon"] = np.float32(horizon)
            X["lead_gap"] = np.float32(horizon - hres_lead(horizon))
            row_parts.append(X)
            hres_u_parts.append(hu)
            hres_v_parts.append(hv)
            y_u_parts.append(yu)
            y_v_parts.append(yv)
    X_all = pd.concat(row_parts, ignore_index=True)
    hres_u = np.concatenate(hres_u_parts).astype("float32")
    hres_v = np.concatenate(hres_v_parts).astype("float32")
    y_u = np.concatenate(y_u_parts).astype("float32")
    y_v = np.concatenate(y_v_parts).astype("float32")
    hres_speed, hres_dir = speed_dir_from_uv(hres_u, hres_v)
    y_speed, y_dir = speed_dir_from_uv(y_u, y_v)
    ok = (
        np.isfinite(hres_u)
        & np.isfinite(hres_v)
        & np.isfinite(y_u)
        & np.isfinite(y_v)
        & np.isfinite(hres_speed)
        & np.isfinite(y_speed)
    )
    X_all = X_all.loc[ok].reset_index(drop=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return X_all, hres_u[ok], hres_v[ok], y_u[ok], y_v[ok], hres_speed[ok], y_speed[ok]


def sample_rows(
    df: pd.DataFrame,
    cube: CubeStore,
    horizon: int,
    group: str,
    levels: Sequence[str],
    val_year: int,
    train_combos: int,
    val_grid_per_anchor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + val_year * 17 + horizon * 101 + (0 if group == "surface" else 1000))
    years = df["time"].dt.year.to_numpy()
    train_candidates = np.flatnonzero(years < val_year)
    latest_needed = df["time"].iloc[train_candidates] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(18, unit="h")
    ok = latest_needed.map(lambda t: pd.Timestamp(t) in cube.time_to_idx).to_numpy()
    train_candidates = train_candidates[ok]
    combos_per_origin = len(levels) * len(HOURS)
    n_train = min(len(train_candidates), max(1, train_combos // combos_per_origin))
    train_rows = np.sort(rng.choice(train_candidates, size=n_train, replace=False))

    val_parts = []
    anchors = pd.to_datetime([f"{val_year}-{mmdd}" for mmdd in ANCHOR_MMDD])
    for anchor in anchors:
        candidates = np.flatnonzero(df["time"].eq(anchor).to_numpy())
        if len(candidates) > val_grid_per_anchor:
            candidates = rng.choice(candidates, size=val_grid_per_anchor, replace=False)
        val_parts.append(np.sort(candidates))
    val_rows = np.sort(np.concatenate(val_parts))
    return train_rows.astype("int64"), val_rows.astype("int64")


def train_lgbm_regression(X: pd.DataFrame, y: np.ndarray, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=63,
        max_depth=8,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def train_lgbm_quantile(X: pd.DataFrame, y: np.ndarray, alpha: float, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def load_reference_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    speed_path = ROOT / "speed_anchor_backtest_candidates.csv"
    dir_path = ROOT / "direction_anchor_backtest_candidates.csv"
    speed = pd.read_csv(speed_path) if speed_path.exists() else pd.DataFrame()
    direction = pd.read_csv(dir_path) if dir_path.exists() else pd.DataFrame()
    return speed, direction


def reference_score(ref: pd.DataFrame, region: str, group: str, horizon: int, candidate: str) -> float:
    if ref.empty:
        return float("nan")
    m = (
        ref["region"].eq(region)
        & ref["group"].eq(group)
        & ref["horizon"].astype(int).eq(int(horizon))
        & ref["candidate"].eq(candidate)
    )
    if not bool(m.any()):
        return float("nan")
    return float(ref.loc[m, "score"].iloc[0])


def evaluate_block(
    region: str,
    group: str,
    horizon: int,
    problems: set[str],
    val_years: Sequence[int],
    train_combos: int,
    val_grid_per_anchor: int,
    seed: int,
    speed_estimators: int,
    direction_estimators: int,
) -> list[dict]:
    levels = GROUP_LEVELS[group]
    hcols = hres_columns(group, levels, [horizon])
    t0 = time.time()
    log(f"[load] {region}/{group}/d{horizon}")
    cube = load_cube(region, group)
    feat = attach_grid_index(load_feature_df(region, hcols), cube, region, group)
    rows = []
    for val_year in val_years:
        train_rows, val_rows = sample_rows(
            feat, cube, horizon, group, levels, int(val_year), train_combos, val_grid_per_anchor, seed
        )
        log(
            f"[cv] {region}/{group}/d{horizon} val={val_year}: "
            f"train_origins={len(train_rows):,} val_origins={len(val_rows):,}"
        )
        X_tr, hu_tr, hv_tr, yu_tr, yv_tr, hs_tr, ys_tr = make_combo_matrix(
            feat, cube, train_rows, region, group, horizon, levels
        )
        X_vl, hu_vl, hv_vl, yu_vl, yv_vl, hs_vl, ys_vl = make_combo_matrix(
            feat, cube, val_rows, region, group, horizon, levels
        )
        _, hdir_vl = speed_dir_from_uv(hu_vl, hv_vl)
        _, ydir_vl = speed_dir_from_uv(yu_vl, yv_vl)

        if "speed" in problems:
            hres_score, hres_hw = best_symmetric_speed_width(ys_vl, hs_vl)
            residual = ys_tr - hs_tr
            m05 = train_lgbm_quantile(X_tr, residual, 0.05, seed + 11 + val_year, speed_estimators)
            m50 = train_lgbm_quantile(X_tr, residual, 0.50, seed + 12 + val_year, speed_estimators)
            m95 = train_lgbm_quantile(X_tr, residual, 0.95, seed + 13 + val_year, speed_estimators)
            r05 = m05.predict(X_vl).astype("float32")
            r50 = m50.predict(X_vl).astype("float32")
            r95 = m95.predict(X_vl).astype("float32")
            q05 = np.maximum(0.0, hs_vl + r05)
            q50 = np.maximum(q05, hs_vl + r50)
            q95 = np.maximum(q50, hs_vl + r95)
            mos_score = winkler_score(ys_vl, q05, q95)
            rows.append(
                {
                    "region": region,
                    "group": group,
                    "horizon": horizon,
                    "problem": "speed",
                    "val_year": val_year,
                    "hres_score": hres_score,
                    "hres_width": hres_hw,
                    "mos_score": mos_score,
                    "mos_width": float(np.nanmean(q95 - q05)),
                    "train_rows": int(len(X_tr)),
                    "val_rows": int(len(X_vl)),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            log(
                f"[speed] {region}/{group}/d{horizon}/{val_year}: "
                f"hres={hres_score:.4f}/hw{hres_hw:.2f} mos={mos_score:.4f}/w{float(np.nanmean(q95-q05)):.2f}"
            )

        if "direction" in problems:
            hres_score, hres_hw = best_direction_width(ydir_vl, hdir_vl)
            mu = train_lgbm_regression(X_tr, yu_tr - hu_tr, seed + 21 + val_year, direction_estimators)
            mv = train_lgbm_regression(X_tr, yv_tr - hv_tr, seed + 22 + val_year, direction_estimators)
            pu = hu_vl + mu.predict(X_vl).astype("float32")
            pv = hv_vl + mv.predict(X_vl).astype("float32")
            _, pred_dir = speed_dir_from_uv(pu, pv)
            mos_score, mos_hw = best_direction_width(ydir_vl, pred_dir)
            rows.append(
                {
                    "region": region,
                    "group": group,
                    "horizon": horizon,
                    "problem": "direction",
                    "val_year": val_year,
                    "hres_score": hres_score,
                    "hres_width": hres_hw,
                    "mos_score": mos_score,
                    "mos_width": mos_hw,
                    "train_rows": int(len(X_tr)),
                    "val_rows": int(len(X_vl)),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            log(
                f"[direction] {region}/{group}/d{horizon}/{val_year}: "
                f"hres={hres_score:.3f}/hw{hres_hw:.1f} mos={mos_score:.3f}/hw{mos_hw:.1f}"
            )

        del X_tr, X_vl
    return rows


def parse_blocks(raw: str) -> list[tuple[str, str, int]]:
    if raw.strip().lower() == "all":
        return [(region, group, horizon) for region in REGIONS for group in GROUP_LEVELS for horizon in HORIZONS]
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        region, group, horizon = part.split(":")
        if region not in REGIONS:
            raise ValueError(f"Bad region in block {part}")
        if group not in GROUP_LEVELS:
            raise ValueError(f"Bad group in block {part}")
        horizon_int = int(horizon.lower().lstrip("d"))
        if horizon_int not in HORIZONS:
            raise ValueError(f"Bad horizon in block {part}")
        out.append((region, group, horizon_int))
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    speed_ref, dir_ref = load_reference_scores()
    out = (
        rows.groupby(["region", "group", "horizon", "problem"], as_index=False)
        .agg(
            hres_mean=("hres_score", "mean"),
            hres_max=("hres_score", "max"),
            mos_mean=("mos_score", "mean"),
            mos_max=("mos_score", "max"),
            hres_width_mean=("hres_width", "mean"),
            mos_width_mean=("mos_width", "mean"),
            train_rows_min=("train_rows", "min"),
            val_rows_min=("val_rows", "min"),
        )
    )
    ref_scores = []
    for _, r in out.iterrows():
        if r["problem"] == "speed":
            ref_scores.append(reference_score(speed_ref, r["region"], r["group"], int(r["horizon"]), "current_model_interval"))
        else:
            ref_scores.append(reference_score(dir_ref, r["region"], r["group"], int(r["horizon"]), "model_by_level"))
    out["current_model_ref_2021"] = ref_scores
    out["best_cv_family"] = np.where(out["mos_mean"] < out["hres_mean"], "mos_residual", "hres_direct")
    out["best_cv_mean"] = np.minimum(out["mos_mean"], out["hres_mean"])
    out["best_cv_max"] = np.where(out["mos_mean"] < out["hres_mean"], out["mos_max"], out["hres_max"])
    out["delta_vs_hres_mean"] = out["mos_mean"] - out["hres_mean"]
    out["delta_best_vs_current_ref"] = out["best_cv_mean"] - out["current_model_ref_2021"]
    out["gate"] = (
        np.isfinite(out["current_model_ref_2021"])
        & (out["best_cv_mean"] + 3.0 < out["current_model_ref_2021"])
        & (out["best_cv_max"] + 8.0 < out["current_model_ref_2021"])
    )
    return out.sort_values(["problem", "region", "group", "horizon"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=DEFAULT_BLOCKS, help="Comma list like north_sea:surface:7,east_china_sea:pressure:7, or all")
    ap.add_argument("--problems", default="speed,direction", help="speed,direction or one of them")
    ap.add_argument("--val-years", default="2020,2021")
    ap.add_argument("--train-combos", type=int, default=80_000)
    ap.add_argument("--val-grid-per-anchor", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--speed-estimators", type=int, default=260)
    ap.add_argument("--direction-estimators", type=int, default=260)
    ap.add_argument("--tag", default="", help="Optional suffix for output CSVs, e.g. pilot_v1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.tag.strip())
    if tag:
        out_by_fold = OUT_DIR / f"cv_mos_residual_framework_{tag}_by_fold.csv"
        out_summary = OUT_DIR / f"cv_mos_residual_framework_{tag}_summary.csv"
    else:
        out_by_fold = OUT_BY_FOLD
        out_summary = OUT_SUMMARY
    blocks = parse_blocks(args.blocks)
    problems = {p.strip().lower() for p in args.problems.split(",") if p.strip()}
    bad = problems - {"speed", "direction"}
    if bad:
        raise ValueError(f"Unknown problems: {sorted(bad)}")
    val_years = [int(y.strip()) for y in args.val_years.split(",") if y.strip()]

    log("CV MOS/residual framework")
    log(f"blocks={blocks}")
    log(f"problems={sorted(problems)} val_years={val_years} train_combos={args.train_combos:,}")
    all_rows: list[dict] = []
    for region, group, horizon in blocks:
        all_rows.extend(
            evaluate_block(
                region=region,
                group=group,
                horizon=horizon,
                problems=problems,
                val_years=val_years,
                train_combos=args.train_combos,
                val_grid_per_anchor=args.val_grid_per_anchor,
                seed=args.seed,
                speed_estimators=args.speed_estimators,
                direction_estimators=args.direction_estimators,
            )
        )

    by_fold = pd.DataFrame(all_rows)
    summary = summarize(by_fold)
    by_fold.to_csv(out_by_fold, index=False)
    summary.to_csv(out_summary, index=False)
    log("")
    log("Summary:")
    log(summary.to_string(index=False))
    log(f"Wrote {out_by_fold}")
    log(f"Wrote {out_summary}")
    gated = summary[summary["gate"]]
    if len(gated):
        log("")
        log("Gated candidates worth turning into an inference branch:")
        log(gated[["region", "group", "horizon", "problem", "best_cv_family", "best_cv_mean", "current_model_ref_2021"]].to_string(index=False))
    else:
        log("")
        log("No block cleared the conservative gate in this run.")


if __name__ == "__main__":
    main()
