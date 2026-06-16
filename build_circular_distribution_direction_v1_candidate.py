from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_final_submission as AUD
import build_dir_error_width_gridlong_v1_candidate as GL
import build_dir_error_width_newsignal_v1_candidate as DEW
import build_dir_interval_newsignal_v1_candidate as DIW
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_direrrw_ecss14push_v1.csv"
GRIDLONG_CACHE = WORK / "gridlong_dir_error_width_v1_rows_s180.parquet"
FEATURE_CACHE = WORK / "feature_rich_newsignal_v1_rows_s180.parquet"
OUT_CSV = WORK / "pred_circdist_dir_v1.csv"
OUT_ZIP = WORK / "sub_circdist_dir_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_circdist_dir_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_circdist_dir_v1_summary.csv"
DECISION_CSV = WORK / "decision_circdist_dir_v1.csv"
MANIFEST = WORK / "manifest_circdist_dir_v1.json"

VAL_YEARS = (2020, 2021)
BASE_COLUMNS = [
    "target_id",
    "display",
    "problem",
    "region",
    "group",
    "level",
    "horizon",
    "hour",
    "origin_year",
    "month",
    "season",
    "latitude",
    "longitude",
    "actual",
    "base_center",
    "base_hw",
    "hres_center",
    "hres_speed",
    "base_hres_delta",
    "hres_dir_sector",
]

TARGET_IDS = (
    "dir_ns_surface_d1",
    "dir_ns_surface_d7",
    "dir_ns_surface_d14",
    "dir_ns_pressure_d1",
    "dir_ns_pressure_d14",
    "dir_ecs_pressure_d1",
    "dir_ecs_pressure_d14",
    "dir_ecs_surface_d14",
)

FEATURE_TARGET_IDS = {
    "dir_ns_surface_d1",
    "dir_ns_pressure_d1",
    "dir_ecs_pressure_d1",
}
GRID_TARGET_IDS = set(TARGET_IDS) - FEATURE_TARGET_IDS

GROUP_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("level_hour", ("level", "hour")),
    ("season_level_hour", ("season", "level", "hour")),
    ("month_level_hour", ("month", "level", "hour")),
    ("sector_level_hour", ("hres_dir_sector", "level", "hour")),
    ("delta_level_hour", ("delta_bin", "level", "hour")),
    ("speed_level_hour", ("speed_bin", "level", "hour")),
)
CENTER_WEIGHTS = (0.25, 0.50, 0.75)
WIDTH_WEIGHTS = (0.00, 0.25, 0.50)
WIDTH_QUANTILES = (0.80, 0.90)
MIN_COUNTS = (120, 300)
MIN_REGIME_ROWS = 80

PUBLIC_CURRENT_V1: dict[str, dict[str, Any]] = {
    "dir_ns_surface_d1": {"ours": 87.0558, "top_best": 87.86, "top_best_name": "JLShen", "gap_to_best": -0.8042},
    "dir_ns_surface_d7": {"ours": 298.5943, "top_best": 256.44, "top_best_name": "sajayrrr", "gap_to_best": 42.1543},
    "dir_ns_surface_d14": {"ours": 325.4191, "top_best": 298.76, "top_best_name": "JLShen", "gap_to_best": 26.6591},
    "dir_ns_pressure_d1": {"ours": 72.2433, "top_best": 68.60, "top_best_name": "sajayrrr", "gap_to_best": 3.6433},
    "dir_ns_pressure_d14": {"ours": 326.7063, "top_best": 300.28, "top_best_name": "Matteo", "gap_to_best": 26.4263},
    "dir_ecs_pressure_d1": {"ours": 93.9026, "top_best": 93.67, "top_best_name": "sajayrrr", "gap_to_best": 0.2326},
    "dir_ecs_pressure_d14": {"ours": 315.4798, "top_best": 285.90, "top_best_name": "sajayrrr", "gap_to_best": 29.5798},
    "dir_ecs_surface_d14": {"ours": 312.6843, "top_best": 303.76, "top_best_name": "Matteo", "gap_to_best": 8.9243},
}


def to_jsonable(obj: Any) -> Any:
    return GL.to_jsonable(obj)


def remove_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP):
        if path.exists():
            path.unlink()


def target_lookup() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    out.update({k: dict(v) for k, v in DIW.TARGET_BY_ID.items()})
    out.update({k: dict(v) for k, v in GL.TARGET_BY_ID.items()})
    return out


TARGET_BY_ID = target_lookup()
TARGETS = tuple(TARGET_BY_ID[tid] for tid in TARGET_IDS)


def signed_circular_error(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((np.asarray(actual, dtype="float64") - np.asarray(pred, dtype="float64") + 180.0) % 360.0) - 180.0


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["level"] = out["level"].astype(str)
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(-1).astype("int16")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(-1).astype("int16")
    out["season"] = out["season"].astype(str).fillna("missing")
    out["hres_dir_sector"] = out["hres_dir_sector"].astype(str).fillna("missing")
    delta = pd.to_numeric(out["base_hres_delta"], errors="coerce").fillna(-1.0)
    speed = pd.to_numeric(out["hres_speed"], errors="coerce").fillna(-1.0)
    out["delta_bin"] = pd.cut(
        delta,
        bins=[-2.0, 0.0, 15.0, 30.0, 60.0, 120.0, 181.0],
        labels=["missing", "d000_015", "d015_030", "d030_060", "d060_120", "d120_180"],
        include_lowest=True,
    ).astype(str)
    out["speed_bin"] = pd.cut(
        speed,
        bins=[-2.0, 0.0, 3.0, 6.0, 9.0, 12.0, 20.0, 200.0],
        labels=["missing", "s000_003", "s003_006", "s006_009", "s009_012", "s012_020", "s020_plus"],
        include_lowest=True,
    ).astype(str)
    out["base_center"] = pd.to_numeric(out["base_center"], errors="coerce").to_numpy(dtype="float64") % 360.0
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce").to_numpy(dtype="float64") % 360.0
    out["base_hw"] = pd.to_numeric(out["base_hw"], errors="coerce")
    if out["base_hw"].isna().any():
        filled = DIW.attach_base_hw(out)
        out["base_hw"] = out["base_hw"].fillna(filled["base_hw"])
    out["base_hw"] = np.clip(out["base_hw"].to_numpy(dtype="float64"), 5.0, 179.9)
    resid = signed_circular_error(out["actual"].to_numpy(dtype="float64"), out["base_center"].to_numpy(dtype="float64"))
    out["residual"] = resid
    out["resid_sin"] = np.sin(np.deg2rad(resid))
    out["resid_cos"] = np.cos(np.deg2rad(resid))
    out["abs_residual"] = np.abs(resid)
    return out


def read_rows() -> pd.DataFrame:
    if not GRIDLONG_CACHE.exists() or not FEATURE_CACHE.exists():
        raise SystemExit("Missing row caches; run gridlong and feature-rich direction branches first.")

    grid = pd.read_parquet(GRIDLONG_CACHE, columns=BASE_COLUMNS)
    grid = grid[grid["target_id"].isin(GRID_TARGET_IDS)].copy()
    feat_columns = [c for c in BASE_COLUMNS if c != "base_hw"]
    feat = pd.read_parquet(FEATURE_CACHE, columns=feat_columns)
    feat["base_hw"] = np.nan
    feat = feat[feat["target_id"].isin(FEATURE_TARGET_IDS)].copy()
    rows = pd.concat([grid, feat], ignore_index=True, sort=False)
    rows = rows[rows["target_id"].isin(TARGET_IDS)].reset_index(drop=True)
    return add_bins(rows)


def circular_group_stats(train: pd.DataFrame, cols: tuple[str, ...], q: float) -> pd.DataFrame:
    agg = (
        train.groupby(list(cols), dropna=False, as_index=False)
        .agg(
            count=("residual", "size"),
            sin_mean=("resid_sin", "mean"),
            cos_mean=("resid_cos", "mean"),
            abs_q=("abs_residual", lambda x: float(np.nanquantile(np.asarray(x, dtype="float64"), q))),
        )
        .reset_index(drop=True)
    )
    agg["bias"] = np.rad2deg(np.arctan2(agg["sin_mean"].to_numpy(dtype="float64"), agg["cos_mean"].to_numpy(dtype="float64")))
    return agg[list(cols) + ["count", "bias", "abs_q"]]


def map_stats(frame: pd.DataFrame, stats: pd.DataFrame, cols: tuple[str, ...], min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keyed = frame[list(cols)].reset_index()
    merged = keyed.merge(stats, on=list(cols), how="left", sort=False).sort_values("index")
    count = pd.to_numeric(merged["count"], errors="coerce").fillna(0).to_numpy(dtype="float64")
    bias = pd.to_numeric(merged["bias"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    width_q = pd.to_numeric(merged["abs_q"], errors="coerce").fillna(np.nan).to_numpy(dtype="float64")
    valid = np.isfinite(width_q) & (count >= float(min_count))
    return bias, width_q, valid


def regime_gain_min(val: pd.DataFrame, pred_center: np.ndarray, pred_hw: np.ndarray) -> float:
    actual = val["actual"].to_numpy(dtype="float64")
    base_center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    gains: list[float] = []
    for cols in (["season"], ["level"], ["hres_dir_sector"], ["level", "hour"]):
        for _, idx in val.groupby(cols, sort=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            if len(idx_arr) < MIN_REGIME_ROWS:
                continue
            base_score = DIW.direction_score_var(actual[idx_arr], base_center[idx_arr], base_hw[idx_arr])
            score = DIW.direction_score_var(actual[idx_arr], pred_center[idx_arr], pred_hw[idx_arr])
            if np.isfinite(base_score) and np.isfinite(score):
                gains.append(float(base_score - score))
    if not gains:
        return float("nan")
    return float(np.nanmin(gains))


def candidate_row(
    target: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    val_year: int,
    group_name: str,
    group_cols: tuple[str, ...],
    center_weight: float,
    width_weight: float,
    width_quantile: float,
    min_count: int,
) -> dict[str, Any]:
    stats = circular_group_stats(train, group_cols, width_quantile)
    bias, width_q, valid = map_stats(val, stats, group_cols, min_count)
    actual = val["actual"].to_numpy(dtype="float64")
    base_center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    pred_center = base_center.copy()
    pred_hw = base_hw.copy()
    pred_center[valid] = (base_center[valid] + center_weight * bias[valid]) % 360.0
    if width_weight > 0:
        raw_hw = np.clip(width_q, 5.0, 179.9)
        pred_hw[valid] = np.clip((1.0 - width_weight) * base_hw[valid] + width_weight * raw_hw[valid], 5.0, 179.9)

    baseline_score = DIW.direction_score_var(actual, base_center, base_hw)
    score = DIW.direction_score_var(actual, pred_center, pred_hw)
    center_move = DIW.circ_abs_diff(pred_center, base_center)
    width_move = np.abs(pred_hw - base_hw)
    changed = (center_move > 0.01) | (width_move > 0.01)
    valid_count = int(valid.sum())
    candidate = f"circdist|grp={group_name}|cw={center_weight:.2f}|ww={width_weight:.2f}|q={width_quantile:.2f}|minn={int(min_count)}"
    return {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "problem": "dir",
        "region": str(target["region"]),
        "group": str(target["group"]),
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "grouping": group_name,
        "center_weight": float(center_weight),
        "width_weight": float(width_weight),
        "width_quantile": float(width_quantile),
        "min_count": int(min_count),
        "score": float(score),
        "baseline_score": float(baseline_score),
        "gain": float(baseline_score - score),
        "regime_gain_min": regime_gain_min(val, pred_center, pred_hw),
        "base_hw_mean": float(np.nanmean(base_hw)),
        "pred_hw_mean": float(np.nanmean(pred_hw)),
        "center_move_mean": float(np.nanmean(center_move)),
        "center_move_p90": float(np.nanquantile(center_move, 0.90)),
        "center_move_p99": float(np.nanquantile(center_move, 0.99)),
        "width_move_mean": float(np.nanmean(width_move)),
        "width_move_p90": float(np.nanquantile(width_move, 0.90)),
        "width_move_p99": float(np.nanquantile(width_move, 0.99)),
        "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "val_selected": valid_count,
        "score_values": int(np.isfinite(actual).sum()),
    }


def baseline_row(target: dict[str, Any], val: pd.DataFrame, val_year: int) -> dict[str, Any]:
    actual = val["actual"].to_numpy(dtype="float64")
    base_center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    score = DIW.direction_score_var(actual, base_center, base_hw)
    return {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "problem": "dir",
        "region": str(target["region"]),
        "group": str(target["group"]),
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": "current_base_row_cache",
        "grouping": "baseline",
        "center_weight": 0.0,
        "width_weight": 0.0,
        "width_quantile": 0.0,
        "min_count": 0,
        "score": float(score),
        "baseline_score": float(score),
        "gain": 0.0,
        "regime_gain_min": 0.0,
        "base_hw_mean": float(np.nanmean(base_hw)),
        "pred_hw_mean": float(np.nanmean(base_hw)),
        "center_move_mean": 0.0,
        "center_move_p90": 0.0,
        "center_move_p99": 0.0,
        "width_move_mean": 0.0,
        "width_move_p90": 0.0,
        "width_move_p99": 0.0,
        "changed_fraction": 0.0,
        "valid_fraction": 1.0,
        "train_rows": 0,
        "val_rows": int(len(val)),
        "val_selected": int(len(val)),
        "score_values": int(np.isfinite(actual).sum()),
    }


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for target in TARGETS:
        tid = str(target["target_id"])
        target_rows = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
        if target_rows.empty:
            print(f"[skip] no rows for {tid}", flush=True)
            continue
        for val_year in VAL_YEARS:
            train = target_rows[target_rows["origin_year"] < int(val_year)].reset_index(drop=True)
            val = target_rows[target_rows["origin_year"].eq(int(val_year))].reset_index(drop=True)
            if train.empty or val.empty:
                continue
            print(f"[cv] {target['display']} y={val_year} train={len(train):,} val={len(val):,}", flush=True)
            out.append(baseline_row(target, val, val_year))
            for group_name, group_cols in GROUP_SPECS:
                for center_weight in CENTER_WEIGHTS:
                    for width_weight in WIDTH_WEIGHTS:
                        for width_quantile in WIDTH_QUANTILES:
                            for min_count in MIN_COUNTS:
                                out.append(
                                    candidate_row(
                                        target,
                                        train,
                                        val,
                                        val_year,
                                        group_name,
                                        group_cols,
                                        center_weight,
                                        width_weight,
                                        width_quantile,
                                        min_count,
                                    )
                                )
    return pd.DataFrame(out)


def gates_for_target(target_id: str, horizon: int) -> dict[str, float]:
    long_horizon = int(horizon) >= 7
    public_gap = float(PUBLIC_CURRENT_V1.get(target_id, {}).get("gap_to_best", 0.0))
    return {
        "min_public_gap": 1.0 if int(horizon) == 1 else 8.0,
        "min_mean_gain": 6.0 if int(horizon) == 1 else 4.0,
        "min_worst_gain": 1.5 if int(horizon) == 1 else 1.0,
        "min_regime_gain": 1.0,
        "min_changed_fraction": 0.03,
        "max_changed_fraction": 0.95,
        "min_valid_fraction": 0.12,
        "max_center_move_p90": 30.0 if not long_horizon else 45.0,
        "max_center_move_p99": 70.0 if not long_horizon else 100.0,
        "max_width_move_p90": 50.0 if not long_horizon else 75.0,
        "max_width_move_p99": 95.0 if not long_horizon else 125.0,
        "min_val_selected": 400 if "surface" in target_id else 1200,
        "public_gap": public_gap,
    }


def summarize_and_gate(folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        folds.groupby(["target_id", "display", "problem", "region", "group", "horizon", "candidate"], as_index=False)
        .agg(
            score=("score", "mean"),
            score_max=("score", "max"),
            baseline_score=("baseline_score", "mean"),
            baseline_score_max=("baseline_score", "max"),
            gain=("gain", "mean"),
            gain_min=("gain", "min"),
            regime_gain_min=("regime_gain_min", "min"),
            center_move_mean=("center_move_mean", "mean"),
            center_move_p90=("center_move_p90", "max"),
            center_move_p99=("center_move_p99", "max"),
            width_move_mean=("width_move_mean", "mean"),
            width_move_p90=("width_move_p90", "max"),
            width_move_p99=("width_move_p99", "max"),
            changed_fraction=("changed_fraction", "mean"),
            valid_fraction=("valid_fraction", "mean"),
            val_selected_min=("val_selected", "min"),
            fold_count=("val_year", "nunique"),
            val_rows_min=("val_rows", "min"),
        )
        .reset_index(drop=True)
    )
    gated: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        tid = str(row["target_id"])
        gates = gates_for_target(tid, int(row["horizon"]))
        public = PUBLIC_CURRENT_V1.get(tid, {"ours": None, "top_best": None, "top_best_name": "", "gap_to_best": 0.0})
        checks = {
            "not_baseline": str(row["candidate"]) != "current_base_row_cache",
            "public_gap": float(public["gap_to_best"]) >= float(gates["min_public_gap"]),
            "mean_gain": float(row["gain"]) >= float(gates["min_mean_gain"]),
            "worst_fold_gain": float(row["gain_min"]) >= float(gates["min_worst_gain"]),
            "regime_gain": np.isfinite(float(row["regime_gain_min"])) and float(row["regime_gain_min"]) >= float(gates["min_regime_gain"]),
            "changed_fraction": float(gates["min_changed_fraction"]) <= float(row["changed_fraction"]) <= float(gates["max_changed_fraction"]),
            "valid_fraction": float(row["valid_fraction"]) >= float(gates["min_valid_fraction"]),
            "center_move_p90": float(row["center_move_p90"]) <= float(gates["max_center_move_p90"]),
            "center_move_p99": float(row["center_move_p99"]) <= float(gates["max_center_move_p99"]),
            "width_move_p90": float(row["width_move_p90"]) <= float(gates["max_width_move_p90"]),
            "width_move_p99": float(row["width_move_p99"]) <= float(gates["max_width_move_p99"]),
            "val_selected": int(row["val_selected_min"]) >= int(gates["min_val_selected"]),
            "fold_count": int(row["fold_count"]) == len(VAL_YEARS),
        }
        out = dict(row)
        out.update(
            {
                "public_current": public["ours"],
                "leader_reference": public["top_best"],
                "public_gap": float(public["gap_to_best"]),
                "top_best_name": str(public.get("top_best_name", "")),
                "gate_passed_cv": bool(all(checks.values())),
                "reject_reasons": ",".join(k for k, ok in checks.items() if not ok),
            }
        )
        for name, ok in checks.items():
            out[f"gate_{name}"] = bool(ok)
        gated.append(out)
    return pd.DataFrame(gated).sort_values(
        ["gate_passed_cv", "gain", "gain_min", "regime_gain_min", "center_move_p90"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )


def parse_candidate(candidate: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for part in candidate.split("|")[1:]:
        key, value = part.split("=", 1)
        params[key] = value
    return {
        "group_name": str(params["grp"]),
        "center_weight": float(params["cw"]),
        "width_weight": float(params["ww"]),
        "width_quantile": float(params["q"]),
        "min_count": int(params["minn"]),
    }


def select_candidates(decisions: pd.DataFrame) -> list[dict[str, Any]]:
    passed = decisions[decisions["gate_passed_cv"].astype(bool)].copy()
    if passed.empty:
        return []
    selected = (
        passed.sort_values(["target_id", "score_max", "gain", "center_move_p90"], ascending=[True, True, False, True], kind="mergesort")
        .groupby("target_id", as_index=False)
        .head(1)
    )
    return selected.to_dict("records")


def apply_policy_to_submission(df: pd.DataFrame, rows: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    tid = str(selected["target_id"])
    target = TARGET_BY_ID[tid]
    params = parse_candidate(str(selected["candidate"]))
    group_cols = dict(GROUP_SPECS)[params["group_name"]]
    train = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
    stats = circular_group_stats(train, group_cols, params["width_quantile"])
    inf = DEW.inference_rows_for_target(df, target)
    inf = add_bins(inf)
    bias, width_q, valid = map_stats(inf, stats, group_cols, params["min_count"])
    idx = inf["_submission_index"].to_numpy(dtype="int64")
    base_center = inf["base_center"].to_numpy(dtype="float64")
    base_hw = inf["base_hw"].to_numpy(dtype="float64")
    pred_center = base_center.copy()
    pred_hw = base_hw.copy()
    pred_center[valid] = (base_center[valid] + params["center_weight"] * bias[valid]) % 360.0
    if params["width_weight"] > 0:
        raw_hw = np.clip(width_q, 5.0, 179.9)
        pred_hw[valid] = np.clip((1.0 - params["width_weight"]) * base_hw[valid] + params["width_weight"] * raw_hw[valid], 5.0, 179.9)
    center_move = DIW.circ_abs_diff(pred_center, base_center)
    width_move = np.abs(pred_hw - base_hw)
    changed = (center_move > 0.01) | (width_move > 0.01)
    gates = gates_for_target(tid, int(target["horizon"]))
    checks = {
        "valid_fraction": float(valid.mean()) >= float(gates["min_valid_fraction"]),
        "changed_fraction": float(gates["min_changed_fraction"]) <= float(changed.mean()) <= float(gates["max_changed_fraction"]),
        "center_move_p90": float(np.nanquantile(center_move, 0.90)) <= float(gates["max_center_move_p90"]),
        "center_move_p99": float(np.nanquantile(center_move, 0.99)) <= float(gates["max_center_move_p99"]),
        "width_move_p90": float(np.nanquantile(width_move, 0.90)) <= float(gates["max_width_move_p90"]),
        "width_move_p99": float(np.nanquantile(width_move, 0.99)) <= float(gates["max_width_move_p99"]),
    }
    audit = {
        "target_id": tid,
        "display": str(target["display"]),
        "candidate": str(selected["candidate"]),
        "rows_in_scope": int(len(idx)),
        "rows_changed": int(changed.sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "center_move_mean": float(np.nanmean(center_move)),
        "center_move_p90": float(np.nanquantile(center_move, 0.90)),
        "center_move_p99": float(np.nanquantile(center_move, 0.99)),
        "width_move_mean": float(np.nanmean(width_move)),
        "width_move_p90": float(np.nanquantile(width_move, 0.90)),
        "width_move_p99": float(np.nanquantile(width_move, 0.99)),
        "inference_gate_passed": bool(all(checks.values())),
        "inference_reject_reasons": ",".join(k for k, ok in checks.items() if not ok),
    }
    if not audit["inference_gate_passed"]:
        return audit
    df.loc[idx, "dir_50"] = pred_center
    df.loc[idx, "dir_05"] = (pred_center - pred_hw) % 360.0
    df.loc[idx, "dir_95"] = (pred_center + pred_hw) % 360.0
    return audit


def station_direction_status() -> dict[str, Any]:
    path = WORK / "decision_stndir_sel_v2.csv"
    if not path.exists():
        return {"status": "not_evaluated_in_this_branch", "reason": "No station direction decision table found."}
    df = pd.read_csv(path)
    passed = int(df.get("gate_passed_cv", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").sum())
    cols = [c for c in ["target", "candidate", "score", "score_max", "gain", "gain_min", "gate_passed_cv", "reject_reasons"] if c in df.columns]
    top = df[cols].head(10).to_dict("records")
    return {
        "status": "blocked_existing_station_direction_probe" if passed == 0 else "existing_probe_has_cv_passes_but_public_family_not_promoted_here",
        "decision_csv": str(path),
        "candidates_passed_cv": passed,
        "top_candidates": top,
        "note": "Station direction is tracked but not emitted by this grid circular-distribution branch.",
    }


def write_manifest(status: str, payload: dict[str, Any]) -> None:
    data = {
        "status": status,
        "mode": "circular_distribution_direction_v1",
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": GL.sha256(BASE_CSV),
        "gridlong_cache": str(GRIDLONG_CACHE),
        "feature_cache": str(FEATURE_CACHE),
        "cv_by_fold_csv": str(CV_BY_FOLD_CSV),
        "cv_summary_csv": str(CV_SUMMARY_CSV),
        "decision_csv": str(DECISION_CSV),
        "out_csv": str(OUT_CSV) if OUT_CSV.exists() else None,
        "out_zip": str(OUT_ZIP) if OUT_ZIP.exists() else None,
        "targets": TARGET_IDS,
        "gate_policy": {
            "val_years": VAL_YEARS,
            "group_specs": GROUP_SPECS,
            "center_weights": CENTER_WEIGHTS,
            "width_weights": WIDTH_WEIGHTS,
            "width_quantiles": WIDTH_QUANTILES,
            "min_counts": MIN_COUNTS,
            "public_current_v1": PUBLIC_CURRENT_V1,
            "public_feedback_use": "aggregate target-level priority and rollback gates only",
        },
        "station_direction_status": station_direction_status(),
        "competition_rule_notes": [
            "Uses only official phase1 training features/reanalysis labels, generated base predictions, and official inference feature files.",
            "No external data, hidden labels, or row-level scoring-server labels are used.",
            "Public leaderboard values are aggregate target-priority and safety gates only.",
            "The branch is fail-closed: no submission is emitted unless mean, worst-fold, regime, movement, and inference gates pass.",
        ],
        "code_hashes": {
            "builder": GL.sha256(Path(__file__).resolve()),
            "base_csv": GL.sha256(BASE_CSV),
        },
    }
    data.update(payload)
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST}", flush=True)


def write_submission(rows: pd.DataFrame, selected: list[dict[str, Any]]) -> dict[str, Any] | None:
    print(f"[submission] reading {BASE_CSV}", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)
    audits = []
    for sel in selected:
        audit = apply_policy_to_submission(df, rows, sel)
        audits.append(audit)
        if not audit.get("inference_gate_passed", False):
            remove_outputs()
            write_manifest(
                "inference_gate_blocked_no_submission",
                {
                    "selected_candidates": selected,
                    "inference_audits": audits,
                },
            )
            return None
    final = E2E.validate_final(df)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None or len(OUT_ZIP.name) >= 64:
        raise SystemExit(f"Bad ZIP: names={names} bad={bad} name_len={len(OUT_ZIP.name)}")
    audit_df = AUD.read_submission_csv(OUT_ZIP)
    validation = AUD.validate(audit_df)
    if not validation["ok"]:
        raise SystemExit(f"Final validation failed: {validation}")
    delta = AUD.diff_against_baseline(audit_df, BASE_CSV)
    output = {
        "zip": str(OUT_ZIP),
        "zip_name_length": int(len(OUT_ZIP.name)),
        "zip_size": int(OUT_ZIP.stat().st_size),
        "csv": str(OUT_CSV),
        "csv_size": int(OUT_CSV.stat().st_size),
        "csv_sha256": GL.sha256(OUT_CSV),
        "zip_sha256": GL.sha256(OUT_ZIP),
        "internal_names": names,
        "internal_csv_size": int(info.file_size),
        "testzip": bad,
        "validation": validation,
        "delta_vs_base": delta,
        "inference_audits": audits,
    }
    return output


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing current best base CSV: {BASE_CSV}. Run .\\run_direrrw_ecss14push_v1_e2e.ps1 first.")
    remove_outputs()
    rows = read_rows()
    print(f"[rows] loaded {len(rows):,} rows across {rows['target_id'].nunique()} targets", flush=True)
    folds = run_cv(rows)
    folds.to_csv(CV_BY_FOLD_CSV, index=False)
    decisions = summarize_and_gate(folds)
    decisions.to_csv(CV_SUMMARY_CSV, index=False)
    decisions.to_csv(DECISION_CSV, index=False)
    print("[cv] top rows:", flush=True)
    print(
        decisions[
            [
                "target_id",
                "candidate",
                "score",
                "score_max",
                "gain",
                "gain_min",
                "regime_gain_min",
                "changed_fraction",
                "valid_fraction",
                "gate_passed_cv",
                "reject_reasons",
            ]
        ]
        .head(18)
        .to_string(index=False),
        flush=True,
    )
    selected = select_candidates(decisions)
    if not selected:
        remove_outputs()
        write_manifest(
            "blocked_no_submission",
            {
                "candidates_evaluated": int(len(decisions)),
                "candidates_passed_cv": 0,
                "top_candidates": decisions.head(24).to_dict("records"),
            },
        )
        return
    output = write_submission(rows, selected)
    if output is None:
        return
    write_manifest(
        "submission_written_after_circular_distribution_direction_gate",
        {
            "candidates_evaluated": int(len(decisions)),
            "candidates_passed_cv": int(decisions["gate_passed_cv"].astype(bool).sum()),
            "selected_candidates": selected,
            "output": output,
        },
    )
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
