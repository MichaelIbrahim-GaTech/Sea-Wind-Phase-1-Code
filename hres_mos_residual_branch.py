#!/usr/bin/env python3
"""
HRES-MOS residual branch for Sea Winds.

This branch trains model-output-statistics residual corrections from the
provided HRES forecasts to the provided historical reanalysis targets.

Compliance notes:
- Uses only official phase1 train/features/inference files already present in
  `runs/v6_pressure_speed/phase1_dataset`.
- Uses 2019-2020 origins for model fitting.
- Uses the eight 2021 same-calendar origins only for validation/gating and
  direction interval calibration.
- Uses official 2022 inference feature/context rows only to generate the final
  submission. It does not use evaluation targets.

The output starts from the current best compact CSV and patches only
grid-direction blocks whose MOS validation score beats the existing direct
model validation baseline by a conservative margin.
"""
from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("runs/v6_pressure_speed")
DATA = ROOT / "phase1_dataset"
FEATURES = DATA / "features"
TRAIN = DATA / "train"

BASE_CSV_CANDIDATES = [
    ROOT / "predictions_ns_station_d14_monthclim_only_compact.csv",
    ROOT / "predictions_public_positive_fullrefit_hybrid_compact.csv",
]
OUT_CSV = ROOT / "predictions_hres_mos_residual_v1_compact.csv"
OUT_ZIP = ROOT / "submission_hres_mos_residual_v1_compact.zip"
SUMMARY_CSV = ROOT / "hres_mos_residual_v1_summary.csv"

REGIONS = ("north_sea", "east_china_sea")
GROUP_LEVELS = {
    "surface": ("10m", "100m"),
    "pressure": ("1000", "925", "850", "700", "500"),
}
HOURS = (0, 6, 12, 18)

# Focus the expensive MOS training on blocks where the earlier validation table
# showed HRES or HRES/model blends had room to beat the direct model.
BLOCKS = (
    ("north_sea", "surface", 7),
    ("north_sea", "surface", 14),
    ("north_sea", "pressure", 7),
    ("north_sea", "pressure", 14),
    ("east_china_sea", "surface", 7),
    ("east_china_sea", "surface", 14),
    ("east_china_sea", "pressure", 7),
    ("east_china_sea", "pressure", 14),
)

TRAIN_COMBO_SAMPLE = 220_000
VAL_GRID_PER_DATE = 600
RANDOM_SEED = 20260522
PATCH_MARGIN_DEG = 5.0
DIR_HW_GRID = np.array(list(range(20, 181, 5)) + [179.9], dtype="float64")

BASE_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "ws10",
    "ws100",
    "wind_shear",
    "msl",
    "t2m",
    "sst",
    "z700",
    "z850",
    "blh",
    "cape",
    "ws10_daily_mean",
    "ws10_daily_max",
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


def schema_names(path: Path) -> List[str]:
    return list(pq.read_schema(path).names)


def hres_lead(horizon: int) -> int:
    return horizon if horizon in (1, 7) else 10


def uv_from_speed_dir(speed: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(direction.astype("float64") % 360.0)
    u = -speed.astype("float64") * np.sin(rad)
    v = -speed.astype("float64") * np.cos(rad)
    return u.astype("float32"), v.astype("float32")


def speed_dir_from_uv(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u.astype("float64") ** 2 + v.astype("float64") ** 2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed.astype("float32"), direction.astype("float32")


def safe_round_dir(a: np.ndarray) -> np.ndarray:
    out = np.round(np.asarray(a, dtype="float64") % 360.0, 3)
    out[out >= 360.0] = 0.0
    return out


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def circular_winkler_score(y: np.ndarray, center: np.ndarray, half_width: float) -> float:
    y = y.astype("float64") % 360.0
    center = center.astype("float64") % 360.0
    ok = np.isfinite(y) & np.isfinite(center)
    y = y[ok]
    center = center[ok]
    lo = (center - half_width) % 360.0
    hi = (center + half_width) % 360.0
    width = (hi - lo) % 360.0
    inside = ((y - lo) % 360.0) <= width
    miss = np.minimum(circ_abs_diff(y, lo), circ_abs_diff(y, hi))
    return float(np.mean(width + 20.0 * miss * (~inside)))


def best_direction_width(y: np.ndarray, center: np.ndarray) -> Tuple[float, float]:
    best_score = float("inf")
    best_hw = 90.0
    for hw in DIR_HW_GRID:
        score = circular_winkler_score(y, center, float(hw))
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    return best_score, best_hw


def inference_origins() -> Dict[int, pd.Timestamp]:
    out = {}
    for wid in range(1, 9):
        meta = json.loads((DATA / "inference" / f"window_{wid}" / "metadata.json").read_text())
        out[wid] = pd.Timestamp(meta["context_end"])
    return out


VAL_ORIGINS = tuple(pd.Timestamp(year=2021, month=t.month, day=t.day) for t in inference_origins().values())


@dataclass
class CubeStore:
    n_grid: int
    latlon: pd.DataFrame
    time_to_idx: Dict[pd.Timestamp, int]
    u: Dict[str, np.ndarray]
    v: Dict[str, np.ndarray]


def load_cube(region: str, group: str) -> CubeStore:
    if group == "surface":
        path = TRAIN / f"reanalysis_{region}_6h.parquet"
        cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
        df = pd.read_parquet(path, columns=cols)
        level_cols = {"10m": ("u10", "v10"), "100m": ("u100", "v100")}
    else:
        path = TRAIN / f"reanalysis_pressure_{region}.parquet"
        cols = ["time", "latitude", "longitude"]
        for level in GROUP_LEVELS["pressure"]:
            cols.extend([f"u_{level}", f"v_{level}"])
        df = pd.read_parquet(path, columns=cols)
        level_cols = {level: (f"u_{level}", f"v_{level}") for level in GROUP_LEVELS["pressure"]}

    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    df = df.sort_values(["time", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
    times = pd.Series(df["time"].unique()).sort_values().to_numpy()
    n_times = len(times)
    n_grid = int(len(df) // n_times)
    latlon = df.loc[: n_grid - 1, ["latitude", "longitude"]].reset_index(drop=True)
    u_arrays: Dict[str, np.ndarray] = {}
    v_arrays: Dict[str, np.ndarray] = {}
    for level, (u_col, v_col) in level_cols.items():
        u_arrays[level] = df[u_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
        v_arrays[level] = df[v_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}
    return CubeStore(n_grid=n_grid, latlon=latlon, time_to_idx=time_to_idx, u=u_arrays, v=v_arrays)


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
        raise RuntimeError(f"{region}/{group} has feature rows missing from target grid:\n{bad}")
    out["grid_idx"] = out["grid_idx"].astype("int32")
    return out


def hres_columns(group: str, levels: Iterable[str], horizons: Iterable[int]) -> List[str]:
    cols: List[str] = []
    for horizon in horizons:
        lead = hres_lead(horizon)
        for hour in HOURS:
            if group == "surface":
                cols += [f"fcst_speed_d{lead}_h{hour}", f"fcst_dir_d{lead}_h{hour}"]
            else:
                for level in levels:
                    cols += [f"fcst_u_{level}_d{lead}_h{hour}", f"fcst_v_{level}_d{lead}_h{hour}"]
    return sorted(set(cols))


def validate_feature_order(df: pd.DataFrame, cube: CubeStore, region: str, group: str) -> None:
    if "grid_idx" not in df.columns:
        raise RuntimeError(f"{region}/{group} feature table has no grid_idx")
    if df["grid_idx"].isna().any():
        raise RuntimeError(f"{region}/{group} feature table has missing grid_idx")


def base_feature_matrix(df: pd.DataFrame, row_idx: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(row_idx)))
    for c in BASE_NUMERIC_FEATURES:
        if c in df.columns:
            out[c] = pd.to_numeric(df.iloc[row_idx][c], errors="coerce").to_numpy(dtype="float32")
    for c in BASE_DIRECTION_FEATURES:
        if c in df.columns:
            a = pd.to_numeric(df.iloc[row_idx][c], errors="coerce").to_numpy(dtype="float64")
            rad = np.deg2rad(a % 360.0)
            out[f"{c}_sin"] = np.sin(rad).astype("float32")
            out[f"{c}_cos"] = np.cos(rad).astype("float32")
    return out


def hres_uv_from_rows(df: pd.DataFrame, row_idx: np.ndarray, group: str, level: str, horizon: int, hour: int) -> Tuple[np.ndarray, np.ndarray]:
    lead = hres_lead(horizon)
    if group == "surface":
        sp = pd.to_numeric(df.iloc[row_idx][f"fcst_speed_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
        dr = pd.to_numeric(df.iloc[row_idx][f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
        return uv_from_speed_dir(sp, dr)
    u = pd.to_numeric(df.iloc[row_idx][f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
    v = pd.to_numeric(df.iloc[row_idx][f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float32")
    return u, v


def target_uv(cube: CubeStore, origin_times: np.ndarray, grid_idx: np.ndarray, level: str, horizon: int, hour: int) -> Tuple[np.ndarray, np.ndarray]:
    fut = pd.to_datetime(origin_times) + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    t_idx = np.array([cube.time_to_idx.get(pd.Timestamp(t), -1) for t in fut], dtype="int32")
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
    group: str,
    horizon: int,
    levels: Tuple[str, ...],
    train_mode: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    ok = np.isfinite(hres_u) & np.isfinite(hres_v) & np.isfinite(y_u) & np.isfinite(y_v)
    X_all = X_all.loc[ok].reset_index(drop=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return X_all, hres_u[ok], hres_v[ok], y_u[ok], y_v[ok]


def train_lgbm(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=360,
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


def sample_origin_rows(df: pd.DataFrame, cube: CubeStore, horizon: int, group: str, levels: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + horizon * 11 + (0 if group == "surface" else 1000))
    years = df["time"].dt.year.to_numpy()
    train_candidates = np.flatnonzero(years <= 2020)
    # Keep only origins whose latest target is inside the target cube.
    latest_needed = df["time"].iloc[train_candidates] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(18, unit="h")
    ok = latest_needed.map(lambda t: pd.Timestamp(t) in cube.time_to_idx).to_numpy()
    train_candidates = train_candidates[ok]

    combos_per_origin_grid = len(levels) * len(HOURS)
    n_origin_train = min(len(train_candidates), max(1, TRAIN_COMBO_SAMPLE // combos_per_origin_grid))
    train_rows = np.sort(rng.choice(train_candidates, size=n_origin_train, replace=False))

    val_rows_parts = []
    for origin in VAL_ORIGINS:
        candidates = np.flatnonzero(df["time"].eq(origin).to_numpy())
        if len(candidates) > VAL_GRID_PER_DATE:
            candidates = rng.choice(candidates, size=VAL_GRID_PER_DATE, replace=False)
        val_rows_parts.append(np.sort(candidates))
    val_rows = np.sort(np.concatenate(val_rows_parts))
    return train_rows.astype("int64"), val_rows.astype("int64")


def load_baselines() -> Dict[Tuple[str, str, int], float]:
    path = ROOT / "direction_anchor_backtest_candidates.csv"
    df = pd.read_csv(path)
    out = {}
    m = df["candidate"].eq("model_by_level")
    for _, r in df.loc[m].iterrows():
        out[(str(r["region"]), str(r["group"]), int(r["horizon"]))] = float(r["score"])
    return out


@dataclass
class BlockModel:
    region: str
    group: str
    horizon: int
    levels: Tuple[str, ...]
    model_u: lgb.LGBMRegressor
    model_v: lgb.LGBMRegressor
    feature_columns: List[str]
    selected_mode: str
    half_width: float
    val_score: float
    baseline_score: float


def fit_block(region: str, group: str, horizon: int, df: pd.DataFrame, cube: CubeStore, baseline: float) -> Tuple[BlockModel | None, dict]:
    levels = GROUP_LEVELS[group]
    train_rows, val_rows = sample_origin_rows(df, cube, horizon, group, levels)
    print(
        f"[{region}/{group}/d{horizon}] train origins={len(train_rows):,}, "
        f"val origins={len(val_rows):,}",
        flush=True,
    )
    X_tr, hu_tr, hv_tr, yu_tr, yv_tr = make_combo_matrix(df, cube, train_rows, group, horizon, levels, train_mode=True)
    X_vl, hu_vl, hv_vl, yu_vl, yv_vl = make_combo_matrix(df, cube, val_rows, group, horizon, levels, train_mode=False)
    du = yu_tr - hu_tr
    dv = yv_tr - hv_tr
    model_u = train_lgbm(X_tr, du, RANDOM_SEED + horizon + (0 if group == "surface" else 200))
    model_v = train_lgbm(X_tr, dv, RANDOM_SEED + horizon + (1 if group == "surface" else 201))

    pu = hu_vl + model_u.predict(X_vl).astype("float32")
    pv = hv_vl + model_v.predict(X_vl).astype("float32")
    _, pred_dir = speed_dir_from_uv(pu, pv)
    _, y_dir = speed_dir_from_uv(yu_vl, yv_vl)
    score, hw = best_direction_width(y_dir, pred_dir)
    hres_score, hres_hw = best_direction_width(y_dir, speed_dir_from_uv(hu_vl, hv_vl)[1])
    if hres_score <= score:
        selected_mode = "hres_direct"
        selected_score = hres_score
        selected_hw = hres_hw
    else:
        selected_mode = "mos_residual"
        selected_score = score
        selected_hw = hw
    accepted = selected_score + PATCH_MARGIN_DEG < baseline
    row = {
        "region": region,
        "group": group,
        "horizon": horizon,
        "baseline_model_by_level": baseline,
        "hres_direct_score": hres_score,
        "hres_direct_half_width": hres_hw,
        "mos_score": score,
        "mos_half_width": hw,
        "selected_mode": selected_mode,
        "selected_score": selected_score,
        "selected_half_width": selected_hw,
        "accepted": bool(accepted),
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_vl)),
    }
    print(
        f"[{region}/{group}/d{horizon}] baseline={baseline:.3f} "
        f"hres={hres_score:.3f}/{hres_hw:.1f} mos={score:.3f}/{hw:.1f} "
        f"selected={selected_mode}:{selected_score:.3f}/{selected_hw:.1f} "
        f"{'ACCEPT' if accepted else 'reject'}",
        flush=True,
    )
    if not accepted:
        return None, row
    return (
        BlockModel(
            region=region,
            group=group,
            horizon=horizon,
            levels=levels,
            model_u=model_u,
            model_v=model_v,
            feature_columns=list(X_tr.columns),
            selected_mode=selected_mode,
            half_width=selected_hw,
            val_score=selected_score,
            baseline_score=baseline,
        ),
        row,
    )


def inference_feature_df(region: str, wid: int, extra_cols: Iterable[str]) -> pd.DataFrame:
    path = FEATURES / f"inference_window_{wid}_{region}.parquet"
    available = set(schema_names(path))
    cols = ["time", "latitude", "longitude"]
    cols += [c for c in BASE_NUMERIC_FEATURES if c in available and c not in cols]
    cols += [c for c in BASE_DIRECTION_FEATURES if c in available and c not in cols]
    cols += [c for c in extra_cols if c in available and c not in cols]
    df = pd.read_parquet(path, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    return df.reset_index(drop=True)


def make_inference_matrix(df: pd.DataFrame, group: str, horizon: int, level: str, hour: int, columns: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    row_idx = np.arange(len(df), dtype="int64")
    hu, hv = hres_uv_from_rows(df, row_idx, group, level, horizon, hour)
    X = base_feature_matrix(df, row_idx)
    hspd, hdir = speed_dir_from_uv(hu, hv)
    X["hres_u"] = hu
    X["hres_v"] = hv
    X["hres_speed"] = hspd
    rad = np.deg2rad(hdir.astype("float64"))
    X["hres_dir_sin"] = np.sin(rad).astype("float32")
    X["hres_dir_cos"] = np.cos(rad).astype("float32")
    level_code = {level_name: i for i, level_name in enumerate(GROUP_LEVELS[group])}[level]
    X["level_code"] = np.float32(level_code)
    X["hour_sin"] = np.float32(math.sin(2.0 * math.pi * hour / 24.0))
    X["hour_cos"] = np.float32(math.cos(2.0 * math.pi * hour / 24.0))
    X["horizon"] = np.float32(horizon)
    X["lead_gap"] = np.float32(horizon - hres_lead(horizon))
    X = X.reindex(columns=columns, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return X, hu, hv


def patch_block(df_sub: pd.DataFrame, model: BlockModel) -> int:
    hres_cols = hres_columns(model.group, model.levels, [model.horizon])
    total = 0
    for wid in range(1, 9):
        inf = inference_feature_df(model.region, wid, hres_cols)
        for level in model.levels:
            for hour in HOURS:
                X, hu, hv = make_inference_matrix(inf, model.group, model.horizon, level, hour, model.feature_columns)
                if model.selected_mode == "mos_residual":
                    pu = hu + model.model_u.predict(X).astype("float32")
                    pv = hv + model.model_v.predict(X).astype("float32")
                else:
                    pu = hu
                    pv = hv
                _, center = speed_dir_from_uv(pu, pv)
                loc = (
                    df_sub["type"].eq("grid")
                    & df_sub["region"].eq(model.region)
                    & df_sub["window"].eq(wid)
                    & df_sub["horizon"].eq(model.horizon)
                    & df_sub["hour"].eq(hour)
                    & df_sub["level"].astype(str).eq(level)
                )
                idx = df_sub.index[loc]
                patch = inf[["latitude", "longitude"]].copy()
                patch["center"] = center
                current_keys = df_sub.loc[idx, ["latitude", "longitude"]].copy()
                merged = current_keys.merge(patch, on=["latitude", "longitude"], how="left", sort=False)
                if len(idx) != len(merged) or merged["center"].isna().any():
                    raise RuntimeError(
                        f"patch row mismatch {model.region}/{model.group}/d{model.horizon}/W{wid}/h{hour}/{level}: "
                        f"{len(idx)} rows vs {len(center)} centers"
                    )
                hw = model.half_width
                ordered_center = merged["center"].to_numpy(dtype="float32")
                df_sub.loc[idx, "dir_50"] = safe_round_dir(ordered_center)
                df_sub.loc[idx, "dir_05"] = safe_round_dir(ordered_center - hw)
                df_sub.loc[idx, "dir_95"] = safe_round_dir(ordered_center + hw)
                total += len(idx)
    return total


def validate_and_zip(df: pd.DataFrame) -> None:
    q = df[["q05", "q50", "q95"]].apply(pd.to_numeric, errors="coerce")
    dirs = df[["dir_05", "dir_50", "dir_95"]].apply(pd.to_numeric, errors="coerce")
    bad_speed = q.isna().any(axis=1) | (q["q05"] < -1e-9) | (q["q50"] < q["q05"]) | (q["q95"] < q["q50"])
    bad_dir = dirs.isna().any(axis=1) | dirs.lt(0).any(axis=1) | dirs.ge(360).any(axis=1)
    grid = df["type"].eq("grid")
    station = df["type"].eq("station")
    grid_dup = df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum()
    station_dup = df.loc[station].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum()
    print(
        f"Validation rows={len(df):,} grid={int(grid.sum()):,} station={int(station.sum()):,} "
        f"bad_speed={int(bad_speed.sum())} bad_dir={int(bad_dir.sum())} "
        f"grid_dup={int(grid_dup)} station_dup={int(station_dup)}",
        flush=True,
    )
    if bad_speed.any() or bad_dir.any() or grid_dup or station_dup:
        raise SystemExit("validation failed")
    df.to_csv(OUT_CSV, index=False)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        info = zf.getinfo("predictions.csv")
    print(f"Wrote {OUT_ZIP} zip_size={OUT_ZIP.stat().st_size:,} csv_size={info.file_size:,}", flush=True)


def main() -> None:
    baselines = load_baselines()
    accepted: List[BlockModel] = []
    summary_rows = []
    by_region_group: Dict[Tuple[str, str], Tuple[pd.DataFrame, CubeStore]] = {}

    for region, group, horizon in BLOCKS:
        key = (region, group)
        if key not in by_region_group:
            levels = GROUP_LEVELS[group]
            hcols = hres_columns(group, levels, [h for r, g, h in BLOCKS if r == region and g == group])
            cube = load_cube(region, group)
            feat = attach_grid_index(load_feature_df(region, hcols), cube, region, group)
            validate_feature_order(feat, cube, region, group)
            by_region_group[key] = (feat, cube)
        feat, cube = by_region_group[key]
        baseline = baselines[(region, group, horizon)]
        model, row = fit_block(region, group, horizon, feat, cube, baseline)
        summary_rows.append(row)
        if model is not None:
            accepted.append(model)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary written: {SUMMARY_CSV}", flush=True)
    print(f"Accepted blocks: {[f'{m.region}/{m.group}/d{m.horizon}' for m in accepted]}", flush=True)
    if not accepted:
        print("No MOS residual block cleared the validation gate; no submission written.", flush=True)
        return

    base = next((p for p in BASE_CSV_CANDIDATES if p.exists()), None)
    if base is None:
        raise FileNotFoundError("No current-best base CSV found")
    print(f"Loading base submission CSV: {base}", flush=True)
    sub = pd.read_csv(base, low_memory=False)
    patch_counts = {}
    for model in accepted:
        n = patch_block(sub, model)
        patch_counts[f"{model.region}_{model.group}_d{model.horizon}_direction"] = n
    print(f"Patch counts: {patch_counts}", flush=True)
    validate_and_zip(sub)


if __name__ == "__main__":
    main()
