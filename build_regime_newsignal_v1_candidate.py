from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import direction_anchor_backtest as DAB
import sea_winds_end_to_end_final as E2E
import speed_anchor_backtest as SAB


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
FEATURES = WORK / "phase1_dataset" / "features"

BASE_CSV = WORK / "pred_ns_sfc14_rg90.csv"
RANK_GAP = WORK / "rank_gap" / "rank_gap_top10_20260614_current.csv"
OUT_CSV = WORK / "pred_regime_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_regsig_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_regime_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_regime_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_regime_newsignal_v1.csv"
MANIFEST = WORK / "manifest_regime_newsignal_v1.json"

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS
HOURS = DAB.HOURS
ANCHOR_YEARS = (2019, 2020, 2021)
VAL_YEARS = (2020, 2021)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
SAMPLE_PER_ANCHOR_DATE = int(os.environ.get("SEA_WINDS_REGIME_NEW_SIGNAL_SAMPLE_PER_DATE", "120"))
ROW_CACHE = WORK / f"regime_newsignal_v1_rows_s{SAMPLE_PER_ANCHOR_DATE}.parquet"
SEED = 20260614

DIRECTION_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "target_id": "dir_ns_pressure_d1",
        "display": "Dir NS Pressure d1",
        "problem": "dir",
        "region": "north_sea",
        "group": "pressure",
        "levels": DAB.PRESSURE_LEVELS,
        "horizon": 1,
        "gates": {
            "min_public_gap": 15.0,
            "min_mean_gain": 1.25,
            "min_worst_gain": 0.25,
            "min_regime_gain": 0.50,
            "max_score_max": 94.0,
            "max_move_p90": 18.0,
            "max_changed_fraction": 0.30,
            "min_changed_fraction": 0.015,
            "min_val_selected": 120,
            "min_train_selected": 120,
            "max_infer_move_mean": 5.0,
            "max_infer_move_p90": 16.0,
        },
    },
    {
        "target_id": "dir_ecs_pressure_d1",
        "display": "Dir ECS Pressure d1",
        "problem": "dir",
        "region": "east_china_sea",
        "group": "pressure",
        "levels": DAB.PRESSURE_LEVELS,
        "horizon": 1,
        "gates": {
            "min_public_gap": 20.0,
            "min_mean_gain": 1.75,
            "min_worst_gain": 0.35,
            "min_regime_gain": 0.75,
            "max_score_max": 126.0,
            "max_move_p90": 18.0,
            "max_changed_fraction": 0.30,
            "min_changed_fraction": 0.015,
            "min_val_selected": 120,
            "min_train_selected": 120,
            "max_infer_move_mean": 5.0,
            "max_infer_move_p90": 16.0,
        },
    },
    {
        "target_id": "dir_ecs_pressure_d14",
        "display": "Dir ECS Pressure d14",
        "problem": "dir",
        "region": "east_china_sea",
        "group": "pressure",
        "levels": DAB.PRESSURE_LEVELS,
        "horizon": 14,
        "gates": {
            "min_public_gap": 20.0,
            "min_mean_gain": 2.50,
            "min_worst_gain": 0.75,
            "min_regime_gain": 1.00,
            "max_score_max": 308.0,
            "max_move_p90": 22.0,
            "max_changed_fraction": 0.30,
            "min_changed_fraction": 0.015,
            "min_val_selected": 120,
            "min_train_selected": 120,
            "max_infer_move_mean": 6.0,
            "max_infer_move_p90": 18.0,
        },
    },
    {
        "target_id": "dir_ns_surface_d1",
        "display": "Dir NS Surface d1",
        "problem": "dir",
        "region": "north_sea",
        "group": "surface",
        "levels": ("10m", "100m"),
        "horizon": 1,
        "gates": {
            "min_public_gap": 15.0,
            "min_mean_gain": 1.50,
            "min_worst_gain": 0.35,
            "min_regime_gain": 0.75,
            "max_score_max": 102.0,
            "max_move_p90": 18.0,
            "max_changed_fraction": 0.30,
            "min_changed_fraction": 0.015,
            "min_val_selected": 100,
            "min_train_selected": 100,
            "max_infer_move_mean": 5.0,
            "max_infer_move_p90": 16.0,
        },
    },
)

SPEED_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "target_id": "speed_ecs_surface_d1",
        "display": "WS ECS Surface d1",
        "problem": "speed",
        "region": "east_china_sea",
        "group": "surface",
        "levels": ("10m", "100m"),
        "horizon": 1,
        "gates": {
            "min_public_gap": 0.20,
            "min_mean_gain": 0.025,
            "min_worst_gain": 0.005,
            "min_regime_gain": 0.030,
            "max_score_max": 5.45,
            "max_move_p90": 1.20,
            "max_changed_fraction": 0.35,
            "min_changed_fraction": 0.015,
            "min_val_selected": 80,
            "min_train_selected": 80,
            "max_infer_move_mean": 0.35,
            "max_infer_move_p90": 1.20,
        },
    },
)

TARGETS = DIRECTION_TARGETS + SPEED_TARGETS
TARGET_BY_ID = {str(t["target_id"]): t for t in TARGETS}

FALLBACK_GAPS = {
    "Dir NS Pressure d1": {"ours": 91.7386, "top_best": 68.60, "gap_to_best": 23.1386, "better_count": 7},
    "Dir ECS Pressure d1": {"ours": 124.5988, "top_best": 93.67, "gap_to_best": 30.9288, "better_count": 6},
    "Dir ECS Pressure d14": {"ours": 315.4798, "top_best": 285.90, "gap_to_best": 29.5798, "better_count": 5},
    "Dir NS Surface d1": {"ours": 107.4627, "top_best": 87.86, "gap_to_best": 19.6027, "better_count": 5},
    "WS ECS Surface d1": {"ours": 4.6443, "top_best": 4.39, "gap_to_best": 0.2543, "better_count": 6},
}

DIR_SELECTORS = (
    "hspd_q1",
    "hspd_q1_or_delta_q4",
    "delta_q4",
    "hspd_q1_delta_q4",
    "month_5",
    "month_7",
    "month_9",
    "month_7_or_9",
    "season_jja",
    "sector_n",
    "sector_e",
    "sector_se",
    "sector_s",
    "sector_nw",
    "level_100m",
    "level_1000",
    "level_925",
    "level_850",
    "month_7_delta_q4",
    "season_jja_delta_q4",
    "month_7_level_850",
    "month_7_level_925",
    "hspd_q1_level_850",
    "hspd_q1_level_925",
    "hspd_q1_sector_se",
    "hspd_q1_sector_s",
)

SPEED_SELECTORS = (
    "month_5",
    "hour_18",
    "month_5_hour_18",
    "delta_q4",
    "hspd_q4",
    "hspd_q4_delta_q4",
    "level_100m",
    "level_100m_delta_q4",
    "month_5_level_100m",
    "month_5_hour_18_delta_q4",
)

DIR_GROUPINGS = ("global", "level")
DIR_WEIGHTS = (0.50, 1.00)
DIR_CAPS = (5.0, 10.0, 22.0)
DIR_MIN_GROUP_N = 60

SPEED_GROUPINGS = ("global",)
SPEED_LO_SCALES = (0.70, 0.85, 1.00, 1.20, 1.50)
SPEED_HI_SCALES = (0.70, 0.85, 1.00, 1.20, 1.50)
SPEED_MIN_GROUP_N = 50


def install_fast_anchor_predictors() -> None:
    """Use the same vectorized inference path for the anchor helper modules."""

    def fast_predict_grid_speed_level(features_df, model_bundle_for_level, calib_for_level):
        rows = []
        lat = np.round(pd.to_numeric(features_df["latitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        lon = np.round(pd.to_numeric(features_df["longitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        n = len(features_df)
        for tgt, bundle in model_bundle_for_level.items():
            horizon = int(tgt.split("_")[1][1:])
            hour = int(tgt.split("_")[2][1:])
            feats = bundle["features"]
            X = features_df.loc[:, feats].fillna(0)
            qlo_lgb = bundle["lgb_lo"].predict(X)
            q50_lgb = bundle["lgb_mid"].predict(X)
            qhi_lgb = bundle["lgb_hi"].predict(X)
            qlo_cb = np.mean([m.predict(X) for m in bundle["cb_lo"]], axis=0) if bundle["cb_lo"] else qlo_lgb
            qhi_cb = np.mean([m.predict(X) for m in bundle["cb_hi"]], axis=0) if bundle["cb_hi"] else qhi_lgb
            cal = calib_for_level[horizon]
            w = cal["w"]
            qlo = w * qlo_cb + (1.0 - w) * qlo_lgb
            qhi = w * qhi_cb + (1.0 - w) * qhi_lgb
            q50 = q50_lgb
            qlo = q50 - cal["k_lo"] * (q50 - qlo)
            qhi = q50 + cal["k_hi"] * (qhi - q50)
            qlo = np.maximum(np.minimum(qlo, q50), 0.0)
            qhi = np.maximum(qhi, q50)
            rows.append(
                pd.DataFrame(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "horizon": np.full(n, horizon, dtype=np.int16),
                        "hour": np.full(n, hour, dtype=np.int8),
                        "q05": qlo.astype("float32"),
                        "q50": np.asarray(q50, dtype="float32"),
                        "q95": qhi.astype("float32"),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def fast_predict_grid_direction_level(features_df, model_bundle_for_level, calib_for_level):
        rows = []
        lat = np.round(pd.to_numeric(features_df["latitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        lon = np.round(pd.to_numeric(features_df["longitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        n = len(features_df)
        for tgt, bundle in model_bundle_for_level.items():
            horizon = int(tgt.split("_")[1][1:])
            hour = int(tgt.split("_")[2][1:])
            feats = bundle["features"]
            X = features_df.loc[:, feats].fillna(0)
            pred_deg = np.degrees(np.arctan2(bundle["sin"].predict(X), bundle["cos"].predict(X))) % 360.0
            half_width = calib_for_level[horizon]["half_width"]
            rows.append(
                pd.DataFrame(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "horizon": np.full(n, horizon, dtype=np.int16),
                        "hour": np.full(n, hour, dtype=np.int8),
                        "dir_05": ((pred_deg - half_width) % 360.0).astype("float32"),
                        "dir_50": np.asarray(pred_deg, dtype="float32"),
                        "dir_95": ((pred_deg + half_width) % 360.0).astype("float32"),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    DAB.SOL.predict_grid_direction_level = fast_predict_grid_direction_level
    SAB.SOL.predict_grid_speed_level = fast_predict_grid_speed_level


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    return obj


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP):
        if path.exists():
            path.unlink()


def anchor_dates(year: int) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{int(year)}-{mmdd}" for mmdd in ANCHOR_MMDD])


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def signed_circ_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0


def direction_sector(deg: np.ndarray) -> np.ndarray:
    labels = np.array(["n", "ne", "e", "se", "s", "sw", "w", "nw"], dtype=object)
    val = np.asarray(deg, dtype="float64")
    idx = np.floor(((val % 360.0) + 22.5) / 45.0).astype("int64") % 8
    out = labels[idx]
    out[~np.isfinite(val)] = "missing"
    return out


def circular_mean_residual(residual: np.ndarray) -> tuple[float, float, int]:
    r = np.asarray(residual, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0, 0.0, 0
    sr = np.sin(np.radians(r)).mean()
    cr = np.cos(np.radians(r)).mean()
    strength = float(np.sqrt(sr * sr + cr * cr))
    bias = float(np.degrees(np.arctan2(sr, cr)))
    return bias, strength, int(len(r))


def group_cols_from_name(name: str) -> list[str]:
    if name == "level":
        return ["level"]
    if name == "hour":
        return ["hour"]
    if name == "level_hour":
        return ["level", "hour"]
    return []


def parse_candidate(candidate: str) -> dict[str, Any]:
    if candidate.startswith("dirbias|"):
        parts = dict(item.split("=", 1) for item in candidate.split("|")[1:])
        return {
            "kind": "dir",
            "selector": parts["selector"],
            "grouping": parts["group"],
            "weight": float(parts["w"]),
            "cap": float(parts["cap"]),
        }
    if candidate.startswith("speedscale|"):
        parts = dict(item.split("=", 1) for item in candidate.split("|")[1:])
        return {
            "kind": "speed",
            "selector": parts["selector"],
            "grouping": parts["group"],
        }
    raise ValueError(f"Cannot parse candidate: {candidate}")


def load_rank_gap() -> dict[str, dict[str, Any]]:
    out = {k: dict(v) for k, v in FALLBACK_GAPS.items()}
    if not RANK_GAP.exists():
        return out
    gap = pd.read_csv(RANK_GAP)
    for row in gap.to_dict("records"):
        display = str(row.get("display", ""))
        if display in out:
            out[display] = {
                "ours": float(row["ours"]),
                "top_best": float(row["top_best"]),
                "gap_to_best": float(row["gap_to_best"]),
                "better_count": int(row["better_count"]),
                "top_best_name": str(row.get("top_best_name", "")),
                "priority": float(row.get("priority", 0.0)),
            }
    return out


def fold_eval_frame(train: pd.DataFrame, year: int) -> pd.DataFrame:
    dates = set(anchor_dates(year))
    ev = train[train["time"].isin(dates)].copy()
    parts = []
    for _, part in ev.groupby("time", sort=True):
        if SAMPLE_PER_ANCHOR_DATE > 0 and len(part) > SAMPLE_PER_ANCHOR_DATE:
            parts.append(part.sample(SAMPLE_PER_ANCHOR_DATE, random_state=SEED + int(year)))
        else:
            parts.append(part)
    if not parts:
        raise SystemExit(f"No anchor rows for year={year}")
    return pd.concat(parts, ignore_index=True)


def hres_speed_from_features(eval_df: pd.DataFrame, level: str, horizon: int, hour: int) -> np.ndarray:
    speed = SAB.forecast_speed_from_features(eval_df, level, horizon, hour)
    if speed is not None:
        return np.asarray(speed, dtype="float64")
    return np.full(len(eval_df), np.nan, dtype="float64")


def common_columns(target: dict[str, Any], eval_df: pd.DataFrame, level: str, hour: int, year: int) -> dict[str, Any]:
    month = eval_df["time"].dt.month.to_numpy(dtype="int16")
    return {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "problem": str(target["problem"]),
        "region": str(target["region"]),
        "group": str(target["group"]),
        "horizon": int(target["horizon"]),
        "origin_year": int(year),
        "origin_time": eval_df["time"].astype(str).to_numpy(dtype=object),
        "month": month,
        "season": np.array([season_from_month(int(m)) for m in month], dtype=object),
        "hour": np.full(len(eval_df), int(hour), dtype="int16"),
        "level": np.full(len(eval_df), str(level), dtype=object),
        "latitude": eval_df["latitude"].to_numpy(dtype="float64"),
        "longitude": eval_df["longitude"].to_numpy(dtype="float64"),
    }


def load_dir_train(region: str, bundle: dict[str, Any]) -> pd.DataFrame:
    cols = set(DAB.needed_feature_columns(bundle))
    speed_cols = set(SAB.needed_feature_columns(SAB.load_speed_bundle(region)))
    cols.update(c for c in speed_cols if c.startswith("fcst_speed_"))
    for h in DAB.HORIZONS:
        for hr in HOURS:
            cols.add(f"dir_d{h}_h{hr}")
    df = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=sorted(cols))
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    return df.reset_index(drop=True)


def load_speed_train(region: str, bundle: dict[str, Any]) -> pd.DataFrame:
    cols = set(SAB.needed_feature_columns(bundle))
    for h in SAB.HORIZONS:
        for hr in HOURS:
            cols.add(f"speed_d{h}_h{hr}")
    df = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=sorted(cols))
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    return df.reset_index(drop=True)


def build_direction_rows() -> pd.DataFrame:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for target in DIRECTION_TARGETS:
        by_region.setdefault(str(target["region"]), []).append(target)

    parts: list[pd.DataFrame] = []
    for region, targets in by_region.items():
        print(f"[rows:dir] loading region={region}", flush=True)
        bundle = DAB.load_dir_bundle(region)
        train = load_dir_train(region, bundle)
        surf100 = DAB.load_surface100_lookup(region)
        pressure = DAB.load_pressure_lookup(region)
        for year in ANCHOR_YEARS:
            eval_df = fold_eval_frame(train, year)
            print(f"[rows:dir] region={region} year={year} anchor_rows={len(eval_df):,}", flush=True)
            centers = DAB.predict_model_centers(eval_df, bundle)
            for target in targets:
                horizon = int(target["horizon"])
                for level in tuple(target["levels"]):
                    for hour in HOURS:
                        actual = DAB.target_direction(eval_df, level, horizon, hour, surf100, pressure)
                        base = centers[level][(horizon, hour)]
                        hres = DAB.forecast_dir_from_features(eval_df, level, horizon, hour)
                        hres_arr = np.asarray(hres, dtype="float64") if hres is not None else np.full(len(eval_df), np.nan)
                        hres_speed = hres_speed_from_features(eval_df, level, horizon, hour)
                        part = pd.DataFrame(common_columns(target, eval_df, level, hour, year))
                        part["actual"] = actual % 360.0
                        part["base_center"] = base % 360.0
                        part["base_lo"] = np.nan
                        part["base_hi"] = np.nan
                        part["hres_center"] = hres_arr % 360.0
                        part["hres_speed"] = hres_speed
                        part["base_hres_delta"] = circ_abs_diff(base, hres_arr)
                        part["hres_dir_sector"] = direction_sector(hres_arr)
                        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_speed_rows() -> pd.DataFrame:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for target in SPEED_TARGETS:
        by_region.setdefault(str(target["region"]), []).append(target)

    parts: list[pd.DataFrame] = []
    for region, targets in by_region.items():
        print(f"[rows:speed] loading region={region}", flush=True)
        bundle = SAB.load_speed_bundle(region)
        train = load_speed_train(region, bundle)
        surf100 = SAB.load_surface100_lookup(region)
        pressure = SAB.load_pressure_lookup(region)
        for year in ANCHOR_YEARS:
            eval_df = fold_eval_frame(train, year)
            print(f"[rows:speed] region={region} year={year} anchor_rows={len(eval_df):,}", flush=True)
            quantiles = SAB.predict_model_quantiles(eval_df, bundle)
            for target in targets:
                horizon = int(target["horizon"])
                for level in tuple(target["levels"]):
                    for hour in HOURS:
                        actual = SAB.target_speed(eval_df, level, horizon, hour, surf100, pressure)
                        q = quantiles[level][(horizon, hour)]
                        mid = q["q50"].to_numpy(dtype="float64")
                        lo = q["q05"].to_numpy(dtype="float64")
                        hi = q["q95"].to_numpy(dtype="float64")
                        hres = SAB.forecast_speed_from_features(eval_df, level, horizon, hour)
                        hres_arr = np.asarray(hres, dtype="float64") if hres is not None else np.full(len(eval_df), np.nan)
                        part = pd.DataFrame(common_columns(target, eval_df, level, hour, year))
                        part["actual"] = actual
                        part["base_center"] = mid
                        part["base_lo"] = lo
                        part["base_hi"] = hi
                        part["hres_center"] = hres_arr
                        part["hres_speed"] = hres_arr
                        part["base_hres_delta"] = np.abs(mid - hres_arr)
                        part["hres_dir_sector"] = "na"
                        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def selector_thresholds(rows: pd.DataFrame) -> dict[str, float]:
    def q(col: str, quant: float, default: float) -> float:
        arr = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(arr) == 0:
            return default
        return float(arr.quantile(quant))

    return {
        "hres_q25": q("hres_speed", 0.25, 0.0),
        "hres_q50": q("hres_speed", 0.50, 0.0),
        "hres_q75": q("hres_speed", 0.75, 0.0),
        "delta_q50": q("base_hres_delta", 0.50, 0.0),
        "delta_q75": q("base_hres_delta", 0.75, 0.0),
    }


def selector_mask(rows: pd.DataFrame, selector: str, thresholds: dict[str, float]) -> np.ndarray:
    hspd = pd.to_numeric(rows["hres_speed"], errors="coerce").to_numpy(dtype="float64")
    delta = pd.to_numeric(rows["base_hres_delta"], errors="coerce").to_numpy(dtype="float64")
    month = pd.to_numeric(rows["month"], errors="coerce").to_numpy(dtype="int16")
    hour = pd.to_numeric(rows["hour"], errors="coerce").to_numpy(dtype="int16")
    level = rows["level"].astype(str).str.lower().to_numpy(dtype=object)
    sector = rows["hres_dir_sector"].astype(str).str.lower().to_numpy(dtype=object)
    season = rows["season"].astype(str).str.upper().to_numpy(dtype=object)

    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(rows), dtype=bool),
        "hspd_q1": hspd <= thresholds["hres_q25"],
        "hspd_q2": hspd <= thresholds["hres_q50"],
        "hspd_q4": hspd >= thresholds["hres_q75"],
        "delta_q4": delta >= thresholds["delta_q75"],
        "delta_q3plus": delta >= thresholds["delta_q50"],
        "hspd_q1_delta_q4": (hspd <= thresholds["hres_q25"]) & (delta >= thresholds["delta_q75"]),
        "hspd_q1_or_delta_q4": (hspd <= thresholds["hres_q25"]) | (delta >= thresholds["delta_q75"]),
        "hspd_q4_delta_q4": (hspd >= thresholds["hres_q75"]) & (delta >= thresholds["delta_q75"]),
        "month_5": month == 5,
        "month_7": month == 7,
        "month_9": month == 9,
        "month_7_or_9": np.isin(month, [7, 9]),
        "season_jja": season == "JJA",
        "season_jja_delta_q4": (season == "JJA") & (delta >= thresholds["delta_q75"]),
        "hour_18": hour == 18,
        "hour_0": hour == 0,
        "month_5_hour_18": (month == 5) & (hour == 18),
        "month_5_hour_18_delta_q4": (month == 5) & (hour == 18) & (delta >= thresholds["delta_q75"]),
        "month_7_delta_q4": (month == 7) & (delta >= thresholds["delta_q75"]),
        "sector_n": sector == "n",
        "sector_e": sector == "e",
        "sector_se": sector == "se",
        "sector_s": sector == "s",
        "sector_nw": sector == "nw",
        "hspd_q1_sector_se": (hspd <= thresholds["hres_q25"]) & (sector == "se"),
        "hspd_q1_sector_s": (hspd <= thresholds["hres_q25"]) & (sector == "s"),
    }
    for lev in ("10m", "100m", "1000", "925", "850", "700", "500"):
        key = f"level_{lev.lower()}"
        masks[key] = level == lev.lower()
    masks["level_100m_delta_q4"] = (level == "100m") & (delta >= thresholds["delta_q75"])
    masks["month_5_level_100m"] = (month == 5) & (level == "100m")
    masks["month_7_level_850"] = (month == 7) & (level == "850")
    masks["month_7_level_925"] = (month == 7) & (level == "925")
    masks["hspd_q1_level_850"] = (hspd <= thresholds["hres_q25"]) & (level == "850")
    masks["hspd_q1_level_925"] = (hspd <= thresholds["hres_q25"]) & (level == "925")
    if selector not in masks:
        raise ValueError(f"Unknown selector: {selector}")
    return np.asarray(masks[selector], dtype=bool) & np.isfinite(hspd) & np.isfinite(delta)


def score_dir(actual: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(actual) & np.isfinite(center)
    if int(good.sum()) < 40:
        return np.nan, np.nan
    score, half_width = DAB.cws(actual[good], center[good])
    return float(score), float(half_width)


def score_speed(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    good = np.isfinite(actual) & np.isfinite(lo) & np.isfinite(hi)
    if int(good.sum()) < 40:
        return np.nan
    return float(SAB.winkler(actual[good], lo[good], hi[good]))


def fit_direction_bias(train: pd.DataFrame, selector: str, grouping: str, thresholds: dict[str, float]) -> dict[str, Any] | None:
    mask = selector_mask(train, selector, thresholds)
    selected = train.loc[mask].copy()
    if len(selected) < 40:
        return None
    residual = signed_circ_diff(selected["actual"].to_numpy(), selected["base_center"].to_numpy())
    global_bias, global_strength, global_n = circular_mean_residual(residual)
    group_cols = group_cols_from_name(grouping)
    group_bias: dict[str, dict[str, float]] = {}
    if group_cols:
        for key, sub in selected.groupby(group_cols, sort=False):
            if len(sub) < DIR_MIN_GROUP_N:
                continue
            res = signed_circ_diff(sub["actual"].to_numpy(), sub["base_center"].to_numpy())
            bias, strength, n = circular_mean_residual(res)
            group_key = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            group_bias[group_key] = {"bias": bias, "strength": strength, "n": n}
    return {
        "selector": selector,
        "grouping": grouping,
        "global_bias": global_bias,
        "global_strength": global_strength,
        "global_n": global_n,
        "group_bias": group_bias,
    }


def group_key_array(rows: pd.DataFrame, grouping: str) -> np.ndarray:
    cols = group_cols_from_name(grouping)
    if not cols:
        return np.array(["global"] * len(rows), dtype=object)
    if len(cols) == 1:
        return rows[cols[0]].astype(str).to_numpy(dtype=object)
    return rows[cols].astype(str).agg("|".join, axis=1).to_numpy(dtype=object)


def apply_direction_bias(
    rows: pd.DataFrame,
    fit: dict[str, Any],
    thresholds: dict[str, float],
    weight: float,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = rows["base_center"].to_numpy(dtype="float64") % 360.0
    center = base.copy()
    move = np.zeros(len(rows), dtype="float64")
    active = selector_mask(rows, str(fit["selector"]), thresholds)
    if not np.any(active):
        return center, move, active

    keys = group_key_array(rows, str(fit["grouping"]))
    shifts = np.full(len(rows), float(fit["global_bias"]), dtype="float64")
    for key, info in fit.get("group_bias", {}).items():
        hit = keys == key
        if np.any(hit):
            shifts[hit] = float(info["bias"])
    shifts = np.clip(float(weight) * shifts, -float(cap), float(cap))
    center[active] = (base[active] + shifts[active]) % 360.0
    move[active] = np.abs(shifts[active])
    return center, move, active


def fit_speed_scales(train: pd.DataFrame, selector: str, grouping: str, thresholds: dict[str, float]) -> dict[str, Any] | None:
    mask = selector_mask(train, selector, thresholds)
    selected = train.loc[mask].copy()
    if len(selected) < 40:
        return None

    def best_for(sub: pd.DataFrame) -> dict[str, float] | None:
        y = sub["actual"].to_numpy(dtype="float64")
        mid = sub["base_center"].to_numpy(dtype="float64")
        base_lo = sub["base_lo"].to_numpy(dtype="float64")
        base_hi = sub["base_hi"].to_numpy(dtype="float64")
        left = np.maximum(0.03, mid - base_lo)
        right = np.maximum(0.03, base_hi - mid)
        base_score = score_speed(y, base_lo, base_hi)
        if not np.isfinite(base_score):
            return None
        best = {"lo_scale": 1.0, "hi_scale": 1.0, "score": base_score, "gain": 0.0, "n": int(len(sub))}
        for lo_scale in SPEED_LO_SCALES:
            for hi_scale in SPEED_HI_SCALES:
                lo = np.maximum(0.0, mid - left * float(lo_scale))
                hi = np.maximum(mid, mid + right * float(hi_scale))
                score = score_speed(y, lo, hi)
                if np.isfinite(score) and score < float(best["score"]):
                    best = {
                        "lo_scale": float(lo_scale),
                        "hi_scale": float(hi_scale),
                        "score": float(score),
                        "gain": float(base_score - score),
                        "n": int(len(sub)),
                    }
        return best

    global_best = best_for(selected)
    if global_best is None:
        return None
    group_cols = group_cols_from_name(grouping)
    group_scales: dict[str, dict[str, float]] = {}
    if group_cols:
        for key, sub in selected.groupby(group_cols, sort=False):
            if len(sub) < SPEED_MIN_GROUP_N:
                continue
            best = best_for(sub)
            if best is None:
                continue
            group_key = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            group_scales[group_key] = best
    return {
        "selector": selector,
        "grouping": grouping,
        "global": global_best,
        "group_scales": group_scales,
    }


def apply_speed_scales(rows: pd.DataFrame, fit: dict[str, Any], thresholds: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = rows["base_center"].to_numpy(dtype="float64")
    base_lo = rows["base_lo"].to_numpy(dtype="float64")
    base_hi = rows["base_hi"].to_numpy(dtype="float64")
    lo = base_lo.copy()
    hi = base_hi.copy()
    move = np.zeros(len(rows), dtype="float64")
    active = selector_mask(rows, str(fit["selector"]), thresholds)
    if not np.any(active):
        return lo, hi, move, active

    keys = group_key_array(rows, str(fit["grouping"]))
    lo_scale = np.full(len(rows), float(fit["global"]["lo_scale"]), dtype="float64")
    hi_scale = np.full(len(rows), float(fit["global"]["hi_scale"]), dtype="float64")
    for key, info in fit.get("group_scales", {}).items():
        hit = keys == key
        if np.any(hit):
            lo_scale[hit] = float(info["lo_scale"])
            hi_scale[hit] = float(info["hi_scale"])

    left = np.maximum(0.03, mid - base_lo)
    right = np.maximum(0.03, base_hi - mid)
    new_lo = np.maximum(0.0, mid - left * lo_scale)
    new_hi = np.maximum(mid, mid + right * hi_scale)
    move[active] = np.maximum(np.abs(new_lo[active] - base_lo[active]), np.abs(new_hi[active] - base_hi[active]))
    lo[active] = new_lo[active]
    hi[active] = new_hi[active]
    return lo, hi, move, active


def evaluate_direction_candidate(
    target: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    selector: str,
    grouping: str,
    weight: float,
    cap: float,
    val_year: int,
) -> dict[str, Any] | None:
    thresholds = selector_thresholds(train)
    fit = fit_direction_bias(train, selector, grouping, thresholds)
    if fit is None:
        return None
    pred, move, active = apply_direction_bias(val, fit, thresholds, weight, cap)
    base = val["base_center"].to_numpy(dtype="float64")
    actual = val["actual"].to_numpy(dtype="float64")
    baseline_score, baseline_hw = score_dir(actual, base)
    score, half_width = score_dir(actual, pred)
    if not (np.isfinite(score) and np.isfinite(baseline_score)):
        return None
    regime_base, _ = score_dir(actual[active], base[active])
    regime_score, _ = score_dir(actual[active], pred[active])
    regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
    candidate = f"dirbias|selector={selector}|group={grouping}|w={weight:.2f}|cap={cap:.1f}"
    train_selected = int(selector_mask(train, selector, thresholds).sum())
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "dir",
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "selector": selector,
        "grouping": grouping,
        "score": float(score),
        "half_width": float(half_width),
        "baseline_score": float(baseline_score),
        "baseline_half_width": float(baseline_hw),
        "gain": float(baseline_score - score),
        "regime_score": float(regime_score) if np.isfinite(regime_score) else np.nan,
        "regime_baseline_score": float(regime_base) if np.isfinite(regime_base) else np.nan,
        "regime_gain": float(regime_gain) if np.isfinite(regime_gain) else np.nan,
        "move_mean": float(np.nanmean(move)),
        "move_p90": float(np.nanquantile(move, 0.90)),
        "move_p99": float(np.nanquantile(move, 0.99)),
        "changed_fraction": float(np.mean(np.round(move, 1) > 0.0)),
        "train_selected": train_selected,
        "val_selected": int(active.sum()),
        "fit_global_bias": float(fit["global_bias"]),
        "fit_global_strength": float(fit["global_strength"]),
        "fit_global_n": int(fit["global_n"]),
        "eval_rows": int(len(val)),
        "scored_values": int(np.isfinite(actual).sum()),
    }


def evaluate_speed_candidate(
    target: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    selector: str,
    grouping: str,
    val_year: int,
) -> dict[str, Any] | None:
    thresholds = selector_thresholds(train)
    fit = fit_speed_scales(train, selector, grouping, thresholds)
    if fit is None:
        return None
    lo, hi, move, active = apply_speed_scales(val, fit, thresholds)
    actual = val["actual"].to_numpy(dtype="float64")
    base_lo = val["base_lo"].to_numpy(dtype="float64")
    base_hi = val["base_hi"].to_numpy(dtype="float64")
    baseline_score = score_speed(actual, base_lo, base_hi)
    score = score_speed(actual, lo, hi)
    if not (np.isfinite(score) and np.isfinite(baseline_score)):
        return None
    regime_base = score_speed(actual[active], base_lo[active], base_hi[active])
    regime_score = score_speed(actual[active], lo[active], hi[active])
    regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
    candidate = f"speedscale|selector={selector}|group={grouping}"
    train_selected = int(selector_mask(train, selector, thresholds).sum())
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "speed",
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "selector": selector,
        "grouping": grouping,
        "score": float(score),
        "half_width": float(np.nanmedian((hi - lo) / 2.0)),
        "baseline_score": float(baseline_score),
        "baseline_half_width": float(np.nanmedian((base_hi - base_lo) / 2.0)),
        "gain": float(baseline_score - score),
        "regime_score": float(regime_score) if np.isfinite(regime_score) else np.nan,
        "regime_baseline_score": float(regime_base) if np.isfinite(regime_base) else np.nan,
        "regime_gain": float(regime_gain) if np.isfinite(regime_gain) else np.nan,
        "move_mean": float(np.nanmean(move)),
        "move_p90": float(np.nanquantile(move, 0.90)),
        "move_p99": float(np.nanquantile(move, 0.99)),
        "changed_fraction": float(np.mean(np.round(move, 2) > 0.0)),
        "train_selected": train_selected,
        "val_selected": int(active.sum()),
        "fit_lo_scale": float(fit["global"]["lo_scale"]),
        "fit_hi_scale": float(fit["global"]["hi_scale"]),
        "fit_global_gain": float(fit["global"]["gain"]),
        "eval_rows": int(len(val)),
        "scored_values": int(np.isfinite(actual).sum()),
    }


def baseline_fold_rows(rows: pd.DataFrame, target: dict[str, Any], val_year: int) -> dict[str, Any]:
    val = rows[(rows["target_id"].eq(str(target["target_id"]))) & (rows["origin_year"].eq(int(val_year)))]
    if target["problem"] == "dir":
        score, hw = score_dir(val["actual"].to_numpy(), val["base_center"].to_numpy())
    else:
        score = score_speed(val["actual"].to_numpy(), val["base_lo"].to_numpy(), val["base_hi"].to_numpy())
        hw = float(np.nanmedian((val["base_hi"].to_numpy() - val["base_lo"].to_numpy()) / 2.0))
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": target["problem"],
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": "current_model",
        "selector": "none",
        "grouping": "none",
        "score": float(score),
        "half_width": float(hw),
        "baseline_score": float(score),
        "baseline_half_width": float(hw),
        "gain": 0.0,
        "regime_score": np.nan,
        "regime_baseline_score": np.nan,
        "regime_gain": np.nan,
        "move_mean": 0.0,
        "move_p90": 0.0,
        "move_p99": 0.0,
        "changed_fraction": 0.0,
        "train_selected": 0,
        "val_selected": 0,
        "eval_rows": int(len(val)),
        "scored_values": int(np.isfinite(val["actual"].to_numpy()).sum()),
    }


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for target in TARGETS:
        tid = str(target["target_id"])
        target_rows = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
        for val_year in VAL_YEARS:
            train = target_rows[target_rows["origin_year"] < int(val_year)].reset_index(drop=True)
            val = target_rows[target_rows["origin_year"].eq(int(val_year))].reset_index(drop=True)
            if train.empty or val.empty:
                continue
            out.append(baseline_fold_rows(target_rows, target, val_year))
            print(f"[cv] {target['display']} val_year={val_year} train={len(train):,} val={len(val):,}", flush=True)
            if target["problem"] == "dir":
                for selector in DIR_SELECTORS:
                    for grouping in DIR_GROUPINGS:
                        for weight in DIR_WEIGHTS:
                            for cap in DIR_CAPS:
                                rec = evaluate_direction_candidate(target, train, val, selector, grouping, weight, cap, val_year)
                                if rec is not None:
                                    out.append(rec)
            else:
                for selector in SPEED_SELECTORS:
                    for grouping in SPEED_GROUPINGS:
                        rec = evaluate_speed_candidate(target, train, val, selector, grouping, val_year)
                        if rec is not None:
                            out.append(rec)
    return pd.DataFrame(out)


def summarize_and_gate(folds: pd.DataFrame, gap_info: dict[str, dict[str, Any]]) -> pd.DataFrame:
    summary = (
        folds.groupby(["target_id", "display", "problem", "region", "group", "horizon", "candidate", "selector", "grouping"], as_index=False)
        .agg(
            score=("score", "mean"),
            score_max=("score", "max"),
            half_width=("half_width", "mean"),
            half_width_max=("half_width", "max"),
            baseline_score=("baseline_score", "mean"),
            baseline_score_max=("baseline_score", "max"),
            gain=("gain", "mean"),
            gain_min=("gain", "min"),
            regime_gain=("regime_gain", "mean"),
            regime_gain_min=("regime_gain", "min"),
            move_mean=("move_mean", "mean"),
            move_p90=("move_p90", "max"),
            move_p99=("move_p99", "max"),
            changed_fraction=("changed_fraction", "mean"),
            train_selected_min=("train_selected", "min"),
            val_selected_min=("val_selected", "min"),
            fold_count=("val_year", "nunique"),
            eval_rows=("eval_rows", "min"),
            scored_values=("scored_values", "min"),
        )
        .reset_index(drop=True)
    )
    rows = []
    for row in summary.to_dict("records"):
        target = TARGET_BY_ID[str(row["target_id"])]
        gates_cfg = target["gates"]
        gap = gap_info.get(str(row["display"]), FALLBACK_GAPS[str(row["display"])])
        gates = {
            "not_baseline": str(row["candidate"]) != "current_model",
            "public_gap": float(gap["gap_to_best"]) >= float(gates_cfg["min_public_gap"]),
            "mean_gain": float(row["gain"]) >= float(gates_cfg["min_mean_gain"]),
            "worst_gain": float(row["gain_min"]) >= float(gates_cfg["min_worst_gain"]),
            "regime_worst_gain": np.isfinite(float(row["regime_gain_min"])) and float(row["regime_gain_min"]) >= float(gates_cfg["min_regime_gain"]),
            "score_ceiling": float(row["score_max"]) <= float(gates_cfg["max_score_max"]),
            "cv_move": float(row["move_p90"]) <= float(gates_cfg["max_move_p90"]),
            "changed_fraction": float(gates_cfg["min_changed_fraction"]) <= float(row["changed_fraction"]) <= float(gates_cfg["max_changed_fraction"]),
            "train_selected": int(row["train_selected_min"]) >= int(gates_cfg["min_train_selected"]),
            "val_selected": int(row["val_selected_min"]) >= int(gates_cfg["min_val_selected"]),
            "fold_count": int(row["fold_count"]) == len(VAL_YEARS),
        }
        out = dict(row)
        out.update(
            {
                "public_current": float(gap["ours"]),
                "leader_reference": float(gap["top_best"]),
                "public_gap": float(gap["gap_to_best"]),
                "better_count": int(gap["better_count"]),
                "top_best_name": str(gap.get("top_best_name", "")),
                "gate_passed_cv": bool(all(gates.values())),
                "reject_reasons": ",".join(k for k, ok in gates.items() if not ok),
            }
        )
        for name, ok in gates.items():
            out[f"gate_{name}"] = bool(ok)
        rows.append(out)
    return pd.DataFrame(rows).sort_values(
        ["gate_passed_cv", "gain", "gain_min", "regime_gain_min", "score_max"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )


def select_candidates(decisions: pd.DataFrame) -> list[dict[str, Any]]:
    passed = decisions[decisions["gate_passed_cv"].astype(bool)].copy()
    if passed.empty:
        return []
    selected = (
        passed.sort_values(["target_id", "score_max", "gain", "move_p90"], ascending=[True, True, False, True], kind="mergesort")
        .groupby("target_id", as_index=False)
        .head(1)
    )
    return selected.to_dict("records")


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    for c in ("window", "horizon", "hour"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    for c in SPEED_COLS + DIR_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_inference_features(region: str, window: int) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES / f"inference_window_{int(window)}_{region}.parquet")
    if "time" not in df.columns:
        df["time"] = pd.NaT
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(2)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(2)
    return df.reset_index(drop=True)


def inference_common(target: dict[str, Any], inf: pd.DataFrame, level: str, hour: int, base_center: np.ndarray, hres_center: np.ndarray, hres_speed: np.ndarray) -> pd.DataFrame:
    month = inf["time"].dt.month.fillna(0).astype("int16").to_numpy()
    part = pd.DataFrame(
        {
            "target_id": str(target["target_id"]),
            "display": str(target["display"]),
            "problem": str(target["problem"]),
            "region": str(target["region"]),
            "group": str(target["group"]),
            "horizon": int(target["horizon"]),
            "origin_year": 9999,
            "month": month,
            "season": np.array([season_from_month(int(m)) if int(m) else "UNK" for m in month], dtype=object),
            "hour": np.full(len(inf), int(hour), dtype="int16"),
            "level": np.full(len(inf), str(level), dtype=object),
            "latitude": inf["latitude"].to_numpy(dtype="float64"),
            "longitude": inf["longitude"].to_numpy(dtype="float64"),
            "actual": np.nan,
            "base_center": base_center,
            "base_lo": np.nan,
            "base_hi": np.nan,
            "hres_center": hres_center,
            "hres_speed": hres_speed,
            "base_hres_delta": circ_abs_diff(base_center, hres_center) if str(target["problem"]) == "dir" else np.abs(base_center - hres_center),
            "hres_dir_sector": direction_sector(hres_center) if str(target["problem"]) == "dir" else "na",
        }
    )
    return part


def target_allowed_mask(base: pd.DataFrame, target: dict[str, Any]) -> np.ndarray:
    return (
        base["type"].eq("grid")
        & base["region"].eq(str(target["region"]))
        & base["horizon"].eq(int(target["horizon"]))
        & base["level"].isin(tuple(target["levels"]))
    ).to_numpy(dtype=bool)


def apply_direction_selected(
    out: pd.DataFrame,
    train_rows: pd.DataFrame,
    selected: dict[str, Any],
    bundle_cache: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = TARGET_BY_ID[str(selected["target_id"])]
    params = parse_candidate(str(selected["candidate"]))
    thresholds = selector_thresholds(train_rows)
    fit = fit_direction_bias(train_rows, str(params["selector"]), str(params["grouping"]), thresholds)
    if fit is None:
        raise SystemExit(f"Could not refit selected direction candidate on all anchors: {selected['candidate']}")

    region = str(target["region"])
    horizon = int(target["horizon"])
    levels = tuple(target["levels"])
    bundle = bundle_cache.setdefault(region, DAB.load_dir_bundle(region))
    half_width = float(np.clip(round(float(selected["half_width"]) / 5.0) * 5.0, 15.0, 179.9))
    move_parts: list[np.ndarray] = []
    changed = 0
    target_rows = 0
    active_rows = 0

    for window in range(1, 9):
        inf = load_inference_features(region, window)
        inf["feature_row"] = np.arange(len(inf), dtype="int32")
        centers = DAB.predict_model_centers(inf, bundle)
        feat_key = inf[["latitude", "longitude", "feature_row"]]
        for level in levels:
            for hour in HOURS:
                idx = out.index[
                    out["type"].eq("grid")
                    & out["region"].eq(region)
                    & out["window"].eq(int(window))
                    & out["horizon"].eq(horizon)
                    & out["hour"].eq(int(hour))
                    & out["level"].eq(str(level))
                ]
                cur = out.loc[idx, ["latitude", "longitude", "dir_50"]].reset_index()
                merged = cur.merge(feat_key, on=["latitude", "longitude"], how="left", validate="one_to_one", sort=False)
                if len(merged) != len(cur) or merged["feature_row"].isna().any():
                    raise SystemExit(f"feature alignment failed for {target['target_id']} window={window} level={level} hour={hour}")
                rows_idx = merged["feature_row"].to_numpy(dtype="int64")
                base_center = pd.to_numeric(merged["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
                hres = DAB.forecast_dir_from_features(inf, str(level), horizon, int(hour))
                hres_arr = np.asarray(hres, dtype="float64") if hres is not None else centers[str(level)][(horizon, int(hour))]
                hres_speed = hres_speed_from_features(inf, str(level), horizon, int(hour))
                row_frame = inference_common(target, inf.iloc[rows_idx].reset_index(drop=True), str(level), int(hour), base_center, hres_arr[rows_idx], hres_speed[rows_idx])
                proposed, move, active = apply_direction_bias(row_frame, fit, thresholds, float(params["weight"]), float(params["cap"]))
                target_rows += len(idx)
                active_rows += int(active.sum())
                changed += int((np.round(move, 1) > 0.0).sum())
                move_parts.append(move)
                out.loc[idx, "dir_50"] = proposed
                out.loc[idx, "dir_05"] = (proposed - half_width) % 360.0
                out.loc[idx, "dir_95"] = (proposed + half_width) % 360.0

    move_all = np.concatenate(move_parts) if move_parts else np.array([], dtype="float64")
    return out, {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "dir",
        "candidate": str(selected["candidate"]),
        "half_width": half_width,
        "target_rows": int(target_rows),
        "active_rows": int(active_rows),
        "changed_rows_before_rounding": int(changed),
        "changed_fraction": float(changed / max(target_rows, 1)),
        "inference_move_mean": float(np.nanmean(move_all)) if len(move_all) else 0.0,
        "inference_move_p90": float(np.nanquantile(move_all, 0.90)) if len(move_all) else 0.0,
        "inference_move_p99": float(np.nanquantile(move_all, 0.99)) if len(move_all) else 0.0,
        "fit": fit,
    }


def apply_speed_selected(
    out: pd.DataFrame,
    train_rows: pd.DataFrame,
    selected: dict[str, Any],
    bundle_cache: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = TARGET_BY_ID[str(selected["target_id"])]
    params = parse_candidate(str(selected["candidate"]))
    thresholds = selector_thresholds(train_rows)
    fit = fit_speed_scales(train_rows, str(params["selector"]), str(params["grouping"]), thresholds)
    if fit is None:
        raise SystemExit(f"Could not refit selected speed candidate on all anchors: {selected['candidate']}")

    region = str(target["region"])
    horizon = int(target["horizon"])
    levels = tuple(target["levels"])
    bundle = bundle_cache.setdefault(region, SAB.load_speed_bundle(region))
    move_parts: list[np.ndarray] = []
    changed = 0
    target_rows = 0
    active_rows = 0

    for window in range(1, 9):
        inf = load_inference_features(region, window)
        inf["feature_row"] = np.arange(len(inf), dtype="int32")
        feat_key = inf[["latitude", "longitude", "feature_row"]]
        _ = SAB.predict_model_quantiles(inf, bundle)
        for level in levels:
            for hour in HOURS:
                idx = out.index[
                    out["type"].eq("grid")
                    & out["region"].eq(region)
                    & out["window"].eq(int(window))
                    & out["horizon"].eq(horizon)
                    & out["hour"].eq(int(hour))
                    & out["level"].eq(str(level))
                ]
                cur = out.loc[idx, ["latitude", "longitude", "q05", "q50", "q95"]].reset_index()
                merged = cur.merge(feat_key, on=["latitude", "longitude"], how="left", validate="one_to_one", sort=False)
                if len(merged) != len(cur) or merged["feature_row"].isna().any():
                    raise SystemExit(f"feature alignment failed for {target['target_id']} window={window} level={level} hour={hour}")
                rows_idx = merged["feature_row"].to_numpy(dtype="int64")
                mid = pd.to_numeric(merged["q50"], errors="coerce").to_numpy(dtype="float64")
                lo0 = pd.to_numeric(merged["q05"], errors="coerce").to_numpy(dtype="float64")
                hi0 = pd.to_numeric(merged["q95"], errors="coerce").to_numpy(dtype="float64")
                hres = SAB.forecast_speed_from_features(inf, str(level), horizon, int(hour))
                hres_arr = np.asarray(hres, dtype="float64") if hres is not None else np.full(len(inf), np.nan)
                row_frame = inference_common(target, inf.iloc[rows_idx].reset_index(drop=True), str(level), int(hour), mid, hres_arr[rows_idx], hres_arr[rows_idx])
                row_frame["base_lo"] = lo0
                row_frame["base_hi"] = hi0
                row_frame["base_hres_delta"] = np.abs(mid - hres_arr[rows_idx])
                lo, hi, move, active = apply_speed_scales(row_frame, fit, thresholds)
                target_rows += len(idx)
                active_rows += int(active.sum())
                changed += int((np.round(move, 2) > 0.0).sum())
                move_parts.append(move)
                out.loc[idx, "q05"] = lo
                out.loc[idx, "q95"] = hi

    move_all = np.concatenate(move_parts) if move_parts else np.array([], dtype="float64")
    return out, {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "speed",
        "candidate": str(selected["candidate"]),
        "target_rows": int(target_rows),
        "active_rows": int(active_rows),
        "changed_rows_before_rounding": int(changed),
        "changed_fraction": float(changed / max(target_rows, 1)),
        "inference_move_mean": float(np.nanmean(move_all)) if len(move_all) else 0.0,
        "inference_move_p90": float(np.nanquantile(move_all, 0.90)) if len(move_all) else 0.0,
        "inference_move_p99": float(np.nanquantile(move_all, 0.99)) if len(move_all) else 0.0,
        "fit": fit,
    }


def inference_gate(audit: dict[str, Any], selected: dict[str, Any]) -> tuple[bool, list[str]]:
    target = TARGET_BY_ID[str(selected["target_id"])]
    gates = target["gates"]
    reasons = []
    if not (float(gates["min_changed_fraction"]) <= float(audit["changed_fraction"]) <= float(gates["max_changed_fraction"])):
        reasons.append("changed_fraction")
    if float(audit["inference_move_mean"]) > float(gates["max_infer_move_mean"]):
        reasons.append("inference_move_mean")
    if float(audit["inference_move_p90"]) > float(gates["max_infer_move_p90"]):
        reasons.append("inference_move_p90")
    if int(audit["active_rows"]) <= 0:
        reasons.append("active_rows")
    return not reasons, reasons


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def validate_delta(before: pd.DataFrame, after: pd.DataFrame, allowed_speed: np.ndarray, allowed_dir: np.ndarray) -> dict[str, Any]:
    after[SPEED_COLS] = after[SPEED_COLS].apply(pd.to_numeric, errors="coerce").clip(lower=0).round(2)
    after["q05"] = after[["q05", "q50"]].min(axis=1).round(2)
    after["q95"] = after[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        after[c] = ((pd.to_numeric(after[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)
    speed_changed = rows_changed(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, after, DIR_COLS, 1, circular=True)
    outside_speed = speed_changed & ~allowed_speed
    outside_dir = dir_changed & ~allowed_dir
    grid = after["type"].eq("grid")
    type_counts = after["type"].value_counts(dropna=False).to_dict()
    missing = int(after[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    bad_speed = int(((after["q05"] > after["q50"]) | (after["q50"] > after["q95"]) | (after[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((after[DIR_COLS] < 0) | (after[DIR_COLS] >= 360) | after[DIR_COLS].isna()).any(axis=1).sum())
    grid_dup = int(after.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(after.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    if len(after) != 3_448_800 or type_counts.get("grid") != 3_447_360 or type_counts.get("station") != 1_440:
        raise SystemExit(f"row/type count validation failed rows={len(after)} counts={type_counts}")
    if missing or bad_speed or bad_dir or grid_dup or station_dup or int(outside_speed.sum()) or int(outside_dir.sum()):
        raise SystemExit(
            f"validation failed missing={missing} bad_speed={bad_speed} bad_dir={bad_dir} "
            f"grid_dup={grid_dup} station_dup={station_dup} outside_speed={int(outside_speed.sum())} outside_dir={int(outside_dir.sum())}"
        )
    return {
        "rows": int(len(after)),
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_speed_rows_changed": int(outside_speed.sum()),
        "non_target_direction_rows_changed": int(outside_dir.sum()),
        "allowed_speed_rows": int(allowed_speed.sum()),
        "allowed_direction_rows": int(allowed_dir.sum()),
        "missing_prediction_rows": missing,
        "bad_speed_rows": bad_speed,
        "bad_direction_rows": bad_dir,
        "grid_duplicate_keys": grid_dup,
        "station_duplicate_keys": station_dup,
    }


def write_submission(after: pd.DataFrame) -> dict[str, Any]:
    tmp_csv = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    tmp_zip = OUT_ZIP.with_suffix(OUT_ZIP.suffix + ".tmp")
    for path in (tmp_csv, tmp_zip):
        if path.exists():
            path.unlink()
    after[COLS].to_csv(tmp_csv, index=False)
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    tmp_csv.replace(OUT_CSV)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(tmp_zip) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip validation failed names={names} bad={bad}")
    if len(OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {OUT_ZIP.name}")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    tmp_zip.replace(OUT_ZIP)
    return {
        "csv": str(OUT_CSV),
        "zip": str(OUT_ZIP),
        "zip_size": int(OUT_ZIP.stat().st_size),
        "predictions_csv_size": int(info.file_size),
        "csv_sha256": sha256(OUT_CSV),
    }


def write_manifest(status: str, payload: dict[str, Any]) -> None:
    data = {
        "status": status,
        "reason": payload.get("reason"),
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": sha256(BASE_CSV),
        "rank_gap_csv": str(RANK_GAP),
        "cv_by_fold_csv": str(CV_BY_FOLD_CSV),
        "cv_summary_csv": str(CV_SUMMARY_CSV),
        "decision_csv": str(DECISION_CSV),
        "out_csv": str(OUT_CSV) if OUT_CSV.exists() else None,
        "out_zip": str(OUT_ZIP) if OUT_ZIP.exists() else None,
        "targets": TARGETS,
        "gate_policy": {
            "anchor_years": list(ANCHOR_YEARS),
            "val_years": list(VAL_YEARS),
            "anchor_mmdd": list(ANCHOR_MMDD),
            "sample_per_anchor_date": SAMPLE_PER_ANCHOR_DATE,
            "direction_candidate": "selector-conditioned circular residual bias with global/level groupings",
            "speed_candidate": "selector-conditioned q05/q95 interval scale with q50 locked",
            "selectors": {
                "direction": list(DIR_SELECTORS),
                "speed": list(SPEED_SELECTORS),
            },
            "public_feedback_use": "leaderboard snapshot is used only for target eligibility and safety thresholds",
        },
        "competition_rule_notes": [
            "Uses official phase1 training features, reanalysis targets, current generated base predictions, and inference features only.",
            "No external data or hidden/scoring-server labels are used.",
            "Public leaderboard values are aggregate gates only and are never row-level training labels or features.",
            "Selectors use inference-observable features only: calendar, level/hour, HRES speed/sector, and model-HRES disagreement.",
            "Submission output is emitted only if a candidate clears mean, worst-fold, regime-slice, score, movement, and row-scope gates.",
        ],
        "code_hashes": {
            "builder": sha256(Path(__file__).resolve()),
            "runner": sha256(ROOT / "run_regime_newsignal_v1_e2e.ps1"),
            "direction_anchor_backtest.py": sha256(ROOT / "direction_anchor_backtest.py"),
            "speed_anchor_backtest.py": sha256(ROOT / "speed_anchor_backtest.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
            "base_csv": sha256(BASE_CSV),
        },
    }
    data.update(payload)
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST}", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    install_fast_anchor_predictors()
    cleanup_outputs()
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing current-best base CSV: {BASE_CSV}")

    gap_info = load_rank_gap()
    if ROW_CACHE.exists():
        print(f"[cache] loading {ROW_CACHE}", flush=True)
        rows = pd.read_parquet(ROW_CACHE)
    else:
        rows = pd.concat([build_direction_rows(), build_speed_rows()], ignore_index=True)
        tmp_cache = ROW_CACHE.with_suffix(ROW_CACHE.suffix + ".tmp")
        if tmp_cache.exists():
            tmp_cache.unlink()
        print(f"[cache] writing {ROW_CACHE}", flush=True)
        rows.to_parquet(tmp_cache, index=False)
        if ROW_CACHE.exists():
            ROW_CACHE.unlink()
        tmp_cache.replace(ROW_CACHE)
    folds = run_cv(rows)
    folds.to_csv(CV_BY_FOLD_CSV, index=False)
    decisions = summarize_and_gate(folds, gap_info)
    decisions.to_csv(CV_SUMMARY_CSV, index=False)
    decisions.to_csv(DECISION_CSV, index=False)
    selected = select_candidates(decisions)

    if not selected:
        top = decisions.groupby("display", sort=False).head(5)
        write_manifest(
            "blocked_no_submission",
            {
                "reason": "No regime-selective candidate cleared strict public-gap, mean-gain, worst-fold, regime-slice, score-ceiling, movement, and coverage gates.",
                "candidates_evaluated": int(len(decisions)),
                "selected": [],
                "top_by_target": top.to_dict("records"),
            },
        )
        return

    base = normalize_base(pd.read_csv(BASE_CSV))
    after = base.copy()
    allowed_speed = np.zeros(len(base), dtype=bool)
    allowed_dir = np.zeros(len(base), dtype=bool)
    dir_cache: dict[str, dict[str, Any]] = {}
    speed_cache: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for sel in selected:
        target = TARGET_BY_ID[str(sel["target_id"])]
        train_rows = rows[rows["target_id"].eq(str(target["target_id"]))].reset_index(drop=True)
        if target["problem"] == "dir":
            after, audit = apply_direction_selected(after, train_rows, sel, dir_cache)
            allowed_dir |= target_allowed_mask(base, target)
        else:
            after, audit = apply_speed_selected(after, train_rows, sel, speed_cache)
            allowed_speed |= target_allowed_mask(base, target)
        passed, reasons = inference_gate(audit, sel)
        audit["inference_gate_passed"] = bool(passed)
        audit["inference_reject_reasons"] = reasons
        audits.append(audit)
        if not passed:
            blocked.append(audit)

    if blocked:
        cleanup_outputs()
        write_manifest(
            "blocked_inference_gate",
            {
                "reason": "At least one CV-passing regime candidate failed inference movement or active-row gates.",
                "selected": selected,
                "inference_audits": audits,
            },
        )
        return

    validation = validate_delta(base, after, allowed_speed, allowed_dir)
    submission = write_submission(after)
    write_manifest(
        "submission_written",
        {
            "reason": "One or more regime-selective new-signal candidates passed all CV and inference gates.",
            "selected": selected,
            "inference_audits": audits,
            "validation": validation,
            "submission": submission,
        },
    )


if __name__ == "__main__":
    main()
