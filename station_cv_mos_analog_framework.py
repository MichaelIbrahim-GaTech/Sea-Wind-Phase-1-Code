#!/usr/bin/env python3
"""
Station-level CV harness for Sea Winds MOS/analog candidates.

This is a development tool, not a leaderboard patcher.

Compliance:
- Uses only official files under runs/v6_pressure_speed/phase1_dataset.
- Uses historical station observations and provided grid/HRES features.
- For each validation fold, learned models use only origins from earlier years.
- Historical analog features never look past the validation anchor date.
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
OUT_DIR = ROOT / "station_cv_mos_analog"

REGIONS = ("north_sea", "east_china_sea")
HORIZONS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")

BASE_FEATURES = [
    "t2m",
    "msl",
    "sshf",
    "z700",
    "z850",
    "sst",
    "blh",
    "cape",
    "ws10",
    "wd10",
    "ws100",
    "wd100",
    "wind_shear",
    "woy_sin",
    "woy_cos",
    "ws10_h6",
    "wd10_h6",
    "msl_h6",
    "t2m_h6",
    "ws10_h18",
    "wd10_h18",
    "msl_h18",
    "t2m_h18",
    "ws10_daily_max",
    "ws10_daily_mean",
    "ws10_daily_range",
    "ws10_lag1d",
    "ws10_lag3d",
    "ws10_lag7d",
    "wd10_lag1d",
    "wd10_lag3d",
    "wd10_lag7d",
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

SPEED_WIDTHS = np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12], dtype="float64")
DIR_WIDTHS = np.array(list(range(20, 181, 5)) + [179.9], dtype="float64")


def log(msg: str) -> None:
    print(msg, flush=True)


def hres_lead(horizon: int) -> int:
    return horizon if horizon in (1, 7) else 10


def schema_names(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def circ_mean_deg(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(np.radians(arr)).mean(), np.cos(np.radians(arr)).mean())) % 360.0)


def speed_winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
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


def best_speed_interval(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype="float64")
    center = np.asarray(center, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(center)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    best_score = float("inf")
    best_width = float("nan")
    for hw in SPEED_WIDTHS:
        lo = np.maximum(0.0, center[ok] - hw)
        hi = center[ok] + hw
        score = speed_winkler(y[ok], lo, hi)
        if score < best_score:
            best_score = score
            best_width = float(hw)
    err = np.abs(y[ok] - center[ok])
    for q in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        hw = float(np.nanquantile(err, q))
        lo = np.maximum(0.0, center[ok] - hw)
        hi = center[ok] + hw
        score = speed_winkler(y[ok], lo, hi)
        if score < best_score:
            best_score = score
            best_width = hw
    return best_score, best_width


def direction_score(y: np.ndarray, center: np.ndarray, half_width: float) -> float:
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


def best_direction_interval(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    best_score = float("inf")
    best_width = float("nan")
    for hw in DIR_WIDTHS:
        score = direction_score(y, center, float(hw))
        if score < best_score:
            best_score = score
            best_width = float(hw)
    return best_score, best_width


def blend_dir(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    ar = np.radians(np.asarray(a, dtype="float64") % 360.0)
    br = np.radians(np.asarray(b, dtype="float64") % 360.0)
    x = (1.0 - w_b) * np.cos(ar) + w_b * np.cos(br)
    y = (1.0 - w_b) * np.sin(ar) + w_b * np.sin(br)
    return np.degrees(np.arctan2(y, x)) % 360.0


def hres_columns() -> list[str]:
    cols = []
    for horizon in HORIZONS:
        lead = hres_lead(horizon)
        for hour in HOURS:
            cols.extend([f"fcst_speed_d{lead}_h{hour}", f"fcst_dir_d{lead}_h{hour}"])
            for level in PRESSURE_LEVELS:
                cols.extend([f"fcst_u_{level}_d{lead}_h{hour}", f"fcst_v_{level}_d{lead}_h{hour}"])
    return sorted(set(cols))


def read_station_meta() -> pd.DataFrame:
    meta = pd.read_csv(DATA / "scoring" / "station_metadata.csv")
    meta["station"] = meta["station"].astype(str)
    for c in ["latitude", "longitude", "nearest_grid_lat", "nearest_grid_lon", "height_m"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["nearest_grid_lat"] = meta["nearest_grid_lat"].round(2)
    meta["nearest_grid_lon"] = meta["nearest_grid_lon"].round(2)
    return meta


def load_station_obs(region: str) -> pd.DataFrame:
    df = pd.read_parquet(TRAIN / f"stations_{region}_6h.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df["station"] = df["station"].astype(str)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360.0
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    return df.sort_values(["station", "time"]).drop_duplicates(["station", "time"], keep="last").reset_index(drop=True)


def load_station_origin_rows(region: str, meta: pd.DataFrame) -> pd.DataFrame:
    meta_r = meta[meta["region"].eq(region)].copy()
    coords = meta_r[["nearest_grid_lat", "nearest_grid_lon"]].drop_duplicates().rename(
        columns={"nearest_grid_lat": "latitude", "nearest_grid_lon": "longitude"}
    )
    path = FEATURES / f"train_{region}.parquet"
    available = set(schema_names(path))
    cols = ["time", "latitude", "longitude"]
    cols += [c for c in BASE_FEATURES if c in available]
    cols += [c for c in hres_columns() if c in available]
    cols = list(dict.fromkeys(cols))
    feat = pd.read_parquet(path, columns=cols)
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = pd.to_numeric(feat["latitude"], errors="coerce").round(2)
    feat["longitude"] = pd.to_numeric(feat["longitude"], errors="coerce").round(2)
    feat = feat.merge(coords, on=["latitude", "longitude"], how="inner")
    rows = meta_r.merge(
        feat,
        left_on=["nearest_grid_lat", "nearest_grid_lon"],
        right_on=["latitude", "longitude"],
        how="inner",
        suffixes=("_station", "_grid"),
    )
    rows = rows.rename(columns={"latitude_station": "station_lat", "longitude_station": "station_lon"})
    rows["station_code"] = rows["station"].astype("category").cat.codes.astype("int16")
    rows["origin_year"] = rows["time"].dt.year.astype("int16")
    rows["origin_month"] = rows["time"].dt.month.astype("int8")
    rows["origin_doy"] = rows["time"].dt.dayofyear.astype("int16")
    rows["origin_doy_sin"] = np.sin(2.0 * np.pi * rows["origin_doy"].astype(float) / 366.0)
    rows["origin_doy_cos"] = np.cos(2.0 * np.pi * rows["origin_doy"].astype(float) / 366.0)
    return rows.reset_index(drop=True)


@dataclass
class StationHistory:
    obs: pd.DataFrame
    lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]]
    recent_speed: dict[tuple[str, int, pd.Timestamp, int], float]
    recent_dir: dict[tuple[str, int, pd.Timestamp, int], float]
    clim_cache: dict[tuple[str, int, int, int, int, str, str], float]


def make_history(obs: pd.DataFrame) -> StationHistory:
    lookup = {}
    for row in obs.itertuples(index=False):
        lookup[(str(row.station), pd.Timestamp(row.time))] = (float(row.speed), float(row.direction) if np.isfinite(row.direction) else np.nan)
    recent_speed: dict[tuple[str, int, pd.Timestamp, int], float] = {}
    recent_dir: dict[tuple[str, int, pd.Timestamp, int], float] = {}
    tmp = obs.copy()
    tmp["date0"] = tmp["time"].dt.normalize()
    tmp["dir_sin"] = np.sin(np.radians(tmp["direction"].astype(float)))
    tmp["dir_cos"] = np.cos(np.radians(tmp["direction"].astype(float)))
    for (station, hour), g in tmp.groupby(["station", "hour"], sort=False):
        g = g.sort_values("time")
        for days in (3, 7, 14):
            rs = g["speed"].rolling(days, min_periods=1).mean().to_numpy(dtype="float64")
            rsi = g["dir_sin"].rolling(days, min_periods=1).mean().to_numpy(dtype="float64")
            rco = g["dir_cos"].rolling(days, min_periods=1).mean().to_numpy(dtype="float64")
            rd = np.degrees(np.arctan2(rsi, rco)) % 360.0
            for date0, sp, direc in zip(g["date0"], rs, rd):
                key = (str(station), int(hour), pd.Timestamp(date0), int(days))
                recent_speed[key] = float(sp)
                recent_dir[key] = float(direc)
    return StationHistory(obs=obs, lookup=lookup, recent_speed=recent_speed, recent_dir=recent_dir, clim_cache={})


def filter_origin_mode(rows: pd.DataFrame, val_years: Sequence[int], mode: str) -> pd.DataFrame:
    if mode == "all":
        return rows
    if mode != "anchors":
        raise ValueError(f"Bad origin mode: {mode}")
    years = sorted(set([2019] + [int(y) for y in val_years]))
    anchors = set(pd.to_datetime([f"{year}-{mmdd}" for year in years for mmdd in ANCHOR_MMDD]))
    return rows[rows["time"].isin(anchors)].reset_index(drop=True)


def lookup_station_value(hist: StationHistory, station: str, t: pd.Timestamp, value: str) -> float:
    got = hist.lookup.get((station, pd.Timestamp(t)))
    if got is None:
        return float("nan")
    return got[0] if value == "speed" else got[1]


def cached_clim(
    hist: StationHistory,
    station: str,
    hour: int,
    origin_year: int,
    target_month: int,
    target_doy: int,
    kind: str,
    value: str,
) -> float:
    key = (station, int(hour), int(origin_year), int(target_month), int(target_doy), kind, value)
    if key in hist.clim_cache:
        return hist.clim_cache[key]
    prior = hist.obs[
        hist.obs["station"].eq(station)
        & hist.obs["hour"].eq(int(hour))
        & hist.obs["time"].dt.year.lt(int(origin_year))
    ]
    if kind == "month":
        sub = prior[prior["month"].eq(int(target_month))]
    elif kind == "doy45":
        dist = np.abs(prior["doy"].astype(int) - int(target_doy))
        dist = np.minimum(dist, 366 - dist)
        sub = prior[dist.le(45)]
    elif kind == "annual":
        sub = prior
    else:
        raise ValueError(kind)
    if len(sub) == 0:
        ans = float("nan")
    elif value == "speed":
        ans = float(sub["speed"].mean())
    else:
        ans = circ_mean_deg(sub["direction"].to_numpy(dtype="float64"))
    hist.clim_cache[key] = ans
    return ans


def add_target_features(base: pd.DataFrame, hist: StationHistory, horizon: int, hour: int, include_climatology: bool = True) -> pd.DataFrame:
    out = base.copy()
    lead = hres_lead(horizon)
    out["horizon"] = np.float32(horizon)
    out["target_hour"] = np.float32(hour)
    out["target_hour_sin"] = np.float32(math.sin(2.0 * math.pi * hour / 24.0))
    out["target_hour_cos"] = np.float32(math.cos(2.0 * math.pi * hour / 24.0))
    out["target_time"] = out["time"] + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    out["target_month"] = out["target_time"].dt.month.astype("int8")
    out["target_doy"] = out["target_time"].dt.dayofyear.astype("int16")
    out["target_doy_sin"] = np.sin(2.0 * np.pi * out["target_doy"].astype(float) / 366.0)
    out["target_doy_cos"] = np.cos(2.0 * np.pi * out["target_doy"].astype(float) / 366.0)

    speed_col = f"fcst_speed_d{lead}_h{hour}"
    dir_col = f"fcst_dir_d{lead}_h{hour}"
    out["hres_speed"] = pd.to_numeric(out.get(speed_col, np.nan), errors="coerce")
    out["hres_dir"] = pd.to_numeric(out.get(dir_col, np.nan), errors="coerce") % 360.0
    out["hres_dir_sin"] = np.sin(np.radians(out["hres_dir"].astype(float)))
    out["hres_dir_cos"] = np.cos(np.radians(out["hres_dir"].astype(float)))

    pressure_speeds = []
    pressure_sins = []
    pressure_coss = []
    for level in PRESSURE_LEVELS:
        u = pd.to_numeric(out.get(f"fcst_u_{level}_d{lead}_h{hour}", np.nan), errors="coerce")
        v = pd.to_numeric(out.get(f"fcst_v_{level}_d{lead}_h{hour}", np.nan), errors="coerce")
        sp = np.sqrt(u * u + v * v)
        dr = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        pressure_speeds.append(sp)
        pressure_sins.append(np.sin(np.radians(dr)))
        pressure_coss.append(np.cos(np.radians(dr)))
    out["hres_pressure_speed_mean"] = pd.concat(pressure_speeds, axis=1).mean(axis=1)
    out["hres_pressure_dir_sin_mean"] = pd.concat(pressure_sins, axis=1).mean(axis=1)
    out["hres_pressure_dir_cos_mean"] = pd.concat(pressure_coss, axis=1).mean(axis=1)

    stations = out["station"].astype(str).to_numpy()
    origins = pd.to_datetime(out["time"]).to_numpy()
    target_times = pd.to_datetime(out["target_time"]).to_numpy()
    y_speed = []
    y_dir = []
    for station, t in zip(stations, target_times):
        y_speed.append(lookup_station_value(hist, station, pd.Timestamp(t), "speed"))
        y_dir.append(lookup_station_value(hist, station, pd.Timestamp(t), "direction"))
    out["y_speed"] = y_speed
    out["y_dir"] = np.asarray(y_dir, dtype="float64") % 360.0

    for lag in (0, 1, 2, 3, 7, 14):
        ss = []
        dd = []
        for station, origin in zip(stations, origins):
            lt = pd.Timestamp(origin) - pd.Timedelta(days=lag) + pd.Timedelta(hours=hour)
            ss.append(lookup_station_value(hist, station, lt, "speed"))
            dd.append(lookup_station_value(hist, station, lt, "direction"))
        out[f"lag{lag}_speed_h"] = ss
        direc = np.asarray(dd, dtype="float64") % 360.0
        out[f"lag{lag}_dir"] = direc
        out[f"lag{lag}_dir_sin"] = np.sin(np.radians(direc))
        out[f"lag{lag}_dir_cos"] = np.cos(np.radians(direc))

    for days in (3, 7, 14):
        sp_vals = []
        dir_vals = []
        for station, origin in zip(stations, origins):
            key = (str(station), int(hour), pd.Timestamp(origin).normalize(), int(days))
            sp_vals.append(hist.recent_speed.get(key, np.nan))
            dir_vals.append(hist.recent_dir.get(key, np.nan))
        out[f"recent{days}_speed"] = sp_vals
        d = np.asarray(dir_vals, dtype="float64") % 360.0
        out[f"recent{days}_dir"] = d
        out[f"recent{days}_dir_sin"] = np.sin(np.radians(d))
        out[f"recent{days}_dir_cos"] = np.cos(np.radians(d))

    if include_climatology:
        month_speed = []
        month_dir = []
        doy_speed = []
        doy_dir = []
        annual_speed = []
        annual_dir = []
        for row in out[["station", "time", "target_month", "target_doy"]].itertuples(index=False):
            station = str(row.station)
            origin_year = pd.Timestamp(row.time).year
            target_month = int(row.target_month)
            target_doy = int(row.target_doy)
            annual_speed.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "annual", "speed"))
            annual_dir.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "annual", "direction"))
            month_speed.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "month", "speed"))
            month_dir.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "month", "direction"))
            doy_speed.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "doy45", "speed"))
            doy_dir.append(cached_clim(hist, station, hour, origin_year, target_month, target_doy, "doy45", "direction"))
        out["annual_speed"] = annual_speed
        out["annual_dir"] = np.asarray(annual_dir, dtype="float64") % 360.0
        out["month_speed"] = month_speed
        out["month_dir"] = np.asarray(month_dir, dtype="float64") % 360.0
        out["doy45_speed"] = doy_speed
        out["doy45_dir"] = np.asarray(doy_dir, dtype="float64") % 360.0
        for c in ("annual_dir", "month_dir", "doy45_dir"):
            out[f"{c}_sin"] = np.sin(np.radians(out[c].astype(float)))
            out[f"{c}_cos"] = np.cos(np.radians(out[c].astype(float)))
    return out


def anchor_mask(df: pd.DataFrame, year: int) -> np.ndarray:
    anchors = set(pd.to_datetime([f"{year}-{mmdd}" for mmdd in ANCHOR_MMDD]))
    return df["time"].isin(anchors).to_numpy()


def numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "time",
        "target_time",
        "station",
        "region",
        "source",
        "y_speed",
        "y_dir",
        "latitude",
        "longitude",
    }
    feats = []
    for c in df.columns:
        if c in exclude or c.startswith("fcst_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, objective: str, seed: int, n_estimators: int, alpha: float | None = None) -> lgb.LGBMRegressor:
    params = dict(
        objective=objective,
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        force_col_wise=True,
    )
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def vector_anen_predict(train: pd.DataFrame, val: pd.DataFrame, k: int, season_w: float) -> np.ndarray:
    preds = np.full(len(val), np.nan, dtype="float64")
    train = train[np.isfinite(train["hres_dir"]) & np.isfinite(train["y_dir"])].copy()
    grouped = {key: g for key, g in train.groupby(["station", "target_hour"], sort=False)}
    for i, row in enumerate(val.itertuples(index=False)):
        hres_now = float(getattr(row, "hres_dir"))
        if not np.isfinite(hres_now):
            continue
        key = (str(getattr(row, "station")), float(getattr(row, "target_hour")))
        hist = grouped.get(key)
        if hist is None or len(hist) < 4:
            continue
        hist_hres = hist["hres_dir"].to_numpy(dtype="float64") % 360.0
        hist_y = hist["y_dir"].to_numpy(dtype="float64") % 360.0
        hist_doy = hist["target_doy"].to_numpy(dtype="float64")
        ddoy = np.abs(hist_doy - float(getattr(row, "target_doy")))
        ddoy = np.minimum(ddoy, 366.0 - ddoy) / 45.0
        dist = circ_abs_diff(hist_hres, hres_now) / 45.0 + season_w * ddoy
        take = np.argsort(dist)[: min(k, len(dist))]
        hx = np.cos(np.radians(hist_hres[take]))
        hy = np.sin(np.radians(hist_hres[take]))
        yx = np.cos(np.radians(hist_y[take]))
        yy = np.sin(np.radians(hist_y[take]))
        tx = np.cos(np.radians(hres_now))
        ty = np.sin(np.radians(hres_now))
        corrected = np.degrees(np.arctan2(ty + (yy - hy), tx + (yx - hx))) % 360.0
        preds[i] = circ_mean_deg(corrected)
    return preds


def evaluate_center_candidates(df: pd.DataFrame, val: pd.DataFrame, region: str, horizon: int, val_year: int, problems: set[str]) -> list[dict]:
    rows = []
    if "speed" in problems:
        candidates = {
            "hres_nearest": val["hres_speed"].to_numpy(dtype="float64"),
            "last_same_hour": val["lag0_speed_h"].to_numpy(dtype="float64"),
            "recent3": val["recent3_speed"].to_numpy(dtype="float64"),
            "recent7": val["recent7_speed"].to_numpy(dtype="float64"),
            "recent14": val["recent14_speed"].to_numpy(dtype="float64"),
            "annual_clim": val["annual_speed"].to_numpy(dtype="float64"),
            "month_clim": val["month_speed"].to_numpy(dtype="float64"),
            "doy45_clim": val["doy45_speed"].to_numpy(dtype="float64"),
        }
        candidates["blend_recent7_doy45_050"] = 0.5 * candidates["recent7"] + 0.5 * candidates["doy45_clim"]
        candidates["blend_recent14_month_050"] = 0.5 * candidates["recent14"] + 0.5 * candidates["month_clim"]
        y = val["y_speed"].to_numpy(dtype="float64")
        for name, pred in candidates.items():
            score, width = best_speed_interval(y, pred)
            rows.append({"region": region, "horizon": horizon, "problem": "speed", "candidate": name, "val_year": val_year, "score": score, "width": width, "n": int(np.isfinite(y).sum())})

    if "direction" in problems:
        candidates = {
            "hres_nearest": val["hres_dir"].to_numpy(dtype="float64"),
            "last_same_hour": val["lag0_dir"].to_numpy(dtype="float64"),
            "recent3": val["recent3_dir"].to_numpy(dtype="float64"),
            "recent7": val["recent7_dir"].to_numpy(dtype="float64"),
            "recent14": val["recent14_dir"].to_numpy(dtype="float64"),
            "annual_clim": val["annual_dir"].to_numpy(dtype="float64"),
            "month_clim": val["month_dir"].to_numpy(dtype="float64"),
            "doy45_clim": val["doy45_dir"].to_numpy(dtype="float64"),
        }
        candidates["blend_recent7_doy45_050"] = blend_dir(candidates["recent7"], candidates["doy45_clim"], 0.5)
        candidates["blend_recent14_month_050"] = blend_dir(candidates["recent14"], candidates["month_clim"], 0.5)
        y = val["y_dir"].to_numpy(dtype="float64")
        for name, pred in candidates.items():
            score, width = best_direction_interval(y, pred)
            rows.append({"region": region, "horizon": horizon, "problem": "direction", "candidate": name, "val_year": val_year, "score": score, "width": width, "n": int(np.isfinite(y).sum())})
    return rows


def evaluate_lgbm(train: pd.DataFrame, val: pd.DataFrame, region: str, horizon: int, val_year: int, problems: set[str], seed: int, n_estimators: int) -> list[dict]:
    rows = []
    feats = numeric_features(train)
    X_tr = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    X_vl = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    if "speed" in problems:
        ok_tr = np.isfinite(train["y_speed"].to_numpy(dtype="float64"))
        ok_vl = np.isfinite(val["y_speed"].to_numpy(dtype="float64"))
        if int(ok_tr.sum()) >= 200 and int(ok_vl.sum()) >= 20:
            y_tr = train.loc[ok_tr, "y_speed"].to_numpy(dtype="float64")
            y_vl = val.loc[ok_vl, "y_speed"].to_numpy(dtype="float64")
            m05 = fit_lgbm(X_tr.loc[ok_tr], y_tr, "quantile", seed + 1 + val_year, n_estimators, 0.05)
            m50 = fit_lgbm(X_tr.loc[ok_tr], y_tr, "quantile", seed + 2 + val_year, n_estimators, 0.50)
            m95 = fit_lgbm(X_tr.loc[ok_tr], y_tr, "quantile", seed + 3 + val_year, n_estimators, 0.95)
            q05 = m05.predict(X_vl.loc[ok_vl]).astype("float64")
            q50 = m50.predict(X_vl.loc[ok_vl]).astype("float64")
            q95 = m95.predict(X_vl.loc[ok_vl]).astype("float64")
            lo = np.maximum(0.0, np.minimum(q05, q50))
            hi = np.maximum(q95, q50)
            rows.append({"region": region, "horizon": horizon, "problem": "speed", "candidate": "lgbm_quantile", "val_year": val_year, "score": speed_winkler(y_vl, lo, hi), "width": float(np.nanmean(hi - lo)), "n": int(ok_vl.sum())})
    if "direction" in problems:
        ok_tr = np.isfinite(train["y_dir"].to_numpy(dtype="float64")) & np.isfinite(train["hres_dir"].to_numpy(dtype="float64"))
        ok_vl = np.isfinite(val["y_dir"].to_numpy(dtype="float64")) & np.isfinite(val["hres_dir"].to_numpy(dtype="float64"))
        if int(ok_tr.sum()) >= 200 and int(ok_vl.sum()) >= 20:
            h_tr = np.radians(train.loc[ok_tr, "hres_dir"].to_numpy(dtype="float64") % 360.0)
            y_tr = np.radians(train.loc[ok_tr, "y_dir"].to_numpy(dtype="float64") % 360.0)
            h_vl = np.radians(val.loc[ok_vl, "hres_dir"].to_numpy(dtype="float64") % 360.0)
            y_vl_deg = val.loc[ok_vl, "y_dir"].to_numpy(dtype="float64") % 360.0
            res_x = np.cos(y_tr) - np.cos(h_tr)
            res_y = np.sin(y_tr) - np.sin(h_tr)
            mx = fit_lgbm(X_tr.loc[ok_tr], res_x, "regression", seed + 4 + val_year, n_estimators)
            my = fit_lgbm(X_tr.loc[ok_tr], res_y, "regression", seed + 5 + val_year, n_estimators)
            px = np.cos(h_vl) + mx.predict(X_vl.loc[ok_vl]).astype("float64")
            py = np.sin(h_vl) + my.predict(X_vl.loc[ok_vl]).astype("float64")
            pred = np.degrees(np.arctan2(py, px)) % 360.0
            score, width = best_direction_interval(y_vl_deg, pred)
            rows.append({"region": region, "horizon": horizon, "problem": "direction", "candidate": "lgbm_hres_vector_residual", "val_year": val_year, "score": score, "width": width, "n": int(ok_vl.sum())})
    return rows


def evaluate_anen(train: pd.DataFrame, val: pd.DataFrame, region: str, horizon: int, val_year: int) -> list[dict]:
    rows = []
    y = val["y_dir"].to_numpy(dtype="float64")
    for k in (10, 20, 40):
        for season_w in (0.0, 0.35, 0.75):
            pred = vector_anen_predict(train, val, k=k, season_w=season_w)
            score, width = best_direction_interval(y, pred)
            rows.append(
                {
                    "region": region,
                    "horizon": horizon,
                    "problem": "direction",
                    "candidate": f"vector_anen_k{k}_sw{season_w:.2f}",
                    "val_year": val_year,
                    "score": score,
                    "width": width,
                    "n": int(np.isfinite(y).sum()),
                }
            )
    return rows


def build_combo(base: pd.DataFrame, hist: StationHistory, horizon: int, include_climatology: bool) -> pd.DataFrame:
    parts = [add_target_features(base, hist, horizon, hour, include_climatology=include_climatology) for hour in HOURS]
    return pd.concat(parts, ignore_index=True)


def evaluate_region(
    region: str,
    base: pd.DataFrame,
    hist: StationHistory,
    horizons: Sequence[int],
    val_years: Sequence[int],
    problems: set[str],
    families: set[str],
    seed: int,
    n_estimators: int,
) -> list[dict]:
    all_rows = []
    for horizon in horizons:
        t0 = time.time()
        df = build_combo(base, hist, horizon, include_climatology=("analog" in families))
        for val_year in val_years:
            val = df[anchor_mask(df, int(val_year)) & df["y_speed"].notna()].copy()
            train = df[df["time"].dt.year.lt(int(val_year)) & df["y_speed"].notna()].copy()
            log(f"[cv] {region}/station/d{horizon} val={val_year}: train={len(train):,} val={len(val):,}")
            if "analog" in families:
                all_rows.extend(evaluate_center_candidates(df, val, region, horizon, int(val_year), problems))
            if "anen" in families and "direction" in problems:
                all_rows.extend(evaluate_anen(train, val, region, horizon, int(val_year)))
            if "lgbm" in families:
                all_rows.extend(evaluate_lgbm(train, val, region, horizon, int(val_year), problems, seed, n_estimators))
        log(f"[done] {region}/station/d{horizon} elapsed={time.time() - t0:.1f}s")
    return all_rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = (
        rows.groupby(["region", "horizon", "problem", "candidate"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            width_mean=("width", "mean"),
            n_min=("n", "min"),
        )
        .sort_values(["region", "problem", "horizon", "score_mean"])
        .reset_index(drop=True)
    )
    best_baseline = []
    gate = []
    for _, r in out.iterrows():
        m = (
            rows["region"].eq(r["region"])
            & rows["horizon"].astype(int).eq(int(r["horizon"]))
            & rows["problem"].eq(r["problem"])
            & rows["candidate"].isin(["hres_nearest", "month_clim", "doy45_clim", "recent14", "blend_recent14_month_050"])
        )
        ref = rows.loc[m].groupby("candidate")["score"].mean().min() if bool(m.any()) else np.nan
        best_baseline.append(float(ref) if np.isfinite(ref) else np.nan)
        margin = 0.50 if r["problem"] == "speed" else 8.0
        max_margin = 1.5 if r["problem"] == "speed" else 15.0
        gate.append(bool(np.isfinite(ref) and r["score_mean"] + margin < ref and r["score_max"] + max_margin < ref))
    out["best_simple_ref_mean"] = best_baseline
    out["delta_vs_simple_ref"] = out["score_mean"] - out["best_simple_ref_mean"]
    out["gate"] = gate
    return out


def parse_csv_values(raw: str, valid: Sequence[int] | Sequence[str]) -> list:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part) if isinstance(valid[0], int) else part)
    bad = [v for v in vals if v not in valid]
    if bad:
        raise ValueError(f"Bad values {bad}; valid={valid}")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="north_sea,east_china_sea")
    ap.add_argument("--horizons", default="1,7,14")
    ap.add_argument("--problems", default="speed,direction")
    ap.add_argument("--families", default="analog,anen,lgbm", help="analog,anen,lgbm")
    ap.add_argument("--val-years", default="2020,2021")
    ap.add_argument("--origin-mode", default="anchors", choices=["anchors", "all"], help="anchors is fast and inference-window-like; all is for slower LGBM work")
    ap.add_argument("--lgb-estimators", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20260525)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = parse_csv_values(args.regions, REGIONS)
    horizons = parse_csv_values(args.horizons, HORIZONS)
    problems = set(parse_csv_values(args.problems, ("speed", "direction")))
    families = set(parse_csv_values(args.families, ("analog", "anen", "lgbm")))
    val_years = [int(y.strip()) for y in args.val_years.split(",") if y.strip()]
    tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.tag.strip()) or "run"

    log("Station CV MOS/analog framework")
    log(f"regions={regions} horizons={horizons} problems={sorted(problems)} families={sorted(families)} val_years={val_years} origin_mode={args.origin_mode}")
    meta = read_station_meta()
    all_rows = []
    for region in regions:
        log(f"[load] {region} station origins/features")
        base = filter_origin_mode(load_station_origin_rows(region, meta), val_years, args.origin_mode)
        hist = make_history(load_station_obs(region))
        all_rows.extend(
            evaluate_region(
                region=region,
                base=base,
                hist=hist,
                horizons=horizons,
                val_years=val_years,
                problems=problems,
                families=families,
                seed=args.seed,
                n_estimators=args.lgb_estimators,
            )
        )

    by_fold = pd.DataFrame(all_rows)
    summary = summarize(by_fold)
    by_fold_path = OUT_DIR / f"station_cv_mos_analog_{tag}_by_fold.csv"
    summary_path = OUT_DIR / f"station_cv_mos_analog_{tag}_summary.csv"
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    log("")
    log("Best candidates by block:")
    best = summary.sort_values(["region", "problem", "horizon", "score_mean"]).groupby(["region", "problem", "horizon"], as_index=False).head(3)
    log(best.to_string(index=False))
    gated = summary[summary["gate"]]
    log(f"Wrote {by_fold_path}")
    log(f"Wrote {summary_path}")
    if len(gated):
        log("")
        log("Gated station candidates worth deeper inference work:")
        log(gated[["region", "horizon", "problem", "candidate", "score_mean", "score_max", "best_simple_ref_mean", "delta_vs_simple_ref"]].to_string(index=False))
    else:
        log("")
        log("No station candidate cleared the conservative gate in this run.")


if __name__ == "__main__":
    main()
