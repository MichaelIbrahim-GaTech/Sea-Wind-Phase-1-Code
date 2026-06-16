from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import build_regime_newsignal_v1_candidate as RNS
import direction_anchor_backtest as DAB
import sea_winds_end_to_end_final as E2E
import speed_anchor_backtest as SAB


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_ns_sfc14_rg90.csv"
ROW_CACHE = WORK / "regime_newsignal_v1_rows_s60.parquet"
OUT_CSV = WORK / "pred_learned_residual_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_lrnres_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_learned_residual_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_learned_residual_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_learned_residual_newsignal_v1.csv"
MANIFEST = WORK / "manifest_learned_residual_newsignal_v1.json"

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS
HOURS = DAB.HOURS
VAL_YEARS = (2020, 2021)
SEED = 20260614

DIR_SELECTORS = (
    "all",
    "hspd_q1",
    "delta_q4",
    "hspd_q1_or_delta_q4",
    "month_7_delta_q4",
    "season_jja_delta_q4",
    "month_7_level_850",
    "month_7_level_925",
    "hspd_q1_level_850",
    "hspd_q1_level_925",
    "sector_s",
    "sector_se",
)
DIR_METHODS = ("residual_uv", "direct_uv")
DIR_WEIGHTS = (0.25, 0.50, 0.75)
DIR_CAPS = (5.0, 10.0, 20.0)
DIR_RAW_GATES = (30.0, 60.0, 180.0)

SPEED_SELECTORS = (
    "all",
    "month_5",
    "hour_18",
    "month_5_hour_18",
    "delta_q4",
    "hspd_q4",
    "level_100m",
    "month_5_hour_18_delta_q4",
)
SPEED_WEIGHTS = (0.25, 0.50, 0.75)
SPEED_SCALES = (0.75, 1.00, 1.25)


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
    return obj


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP):
        if path.exists():
            path.unlink()


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def signed_circ_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0


def angle_to_xy(deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.radians(np.asarray(deg, dtype="float64") % 360.0)
    return np.cos(rad), np.sin(rad)


def xy_to_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(y, x)) % 360.0


def parse_candidate(candidate: str) -> dict[str, Any]:
    parts = dict(item.split("=", 1) for item in str(candidate).split("|")[1:])
    if candidate.startswith("dirlrn|"):
        return {
            "problem": "dir",
            "method": parts["method"],
            "selector": parts["selector"],
            "weight": float(parts["w"]),
            "cap": float(parts["cap"]),
            "gate": float(parts["gate"]),
        }
    if candidate.startswith("speedlrn|"):
        return {
            "problem": "speed",
            "selector": parts["selector"],
            "weight": float(parts["w"]),
            "scale": float(parts["scale"]),
        }
    raise ValueError(f"Cannot parse candidate: {candidate}")


def make_features(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    month = pd.to_numeric(df["month"], errors="coerce").fillna(0).to_numpy(dtype="float64")
    hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0).to_numpy(dtype="float64")
    lat = pd.to_numeric(df["latitude"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    lon = pd.to_numeric(df["longitude"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    base_s = pd.to_numeric(df["base_center"], errors="coerce").fillna(0.0)
    hres_s = pd.to_numeric(df["hres_center"], errors="coerce").fillna(base_s)
    base = base_s.to_numpy(dtype="float64")
    hres = hres_s.to_numpy(dtype="float64")
    delta = pd.to_numeric(df["base_hres_delta"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    hspd = pd.to_numeric(df["hres_speed"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    problem = str(df["problem"].iloc[0])

    out["lat"] = lat
    out["lon"] = lon
    out["lat2"] = lat * lat
    out["lon2"] = lon * lon
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["hres_speed"] = hspd
    out["base_hres_delta"] = delta
    out["delta_log1p"] = np.log1p(np.maximum(0.0, delta))

    if problem == "dir":
        bx, by = angle_to_xy(base)
        hx, hy = angle_to_xy(hres)
        out["base_x"] = bx
        out["base_y"] = by
        out["hres_x"] = hx
        out["hres_y"] = hy
        diff = signed_circ_diff(hres, base)
        out["hres_minus_base_sin"] = np.sin(np.radians(diff))
        out["hres_minus_base_cos"] = np.cos(np.radians(diff))
    else:
        lo = pd.to_numeric(df["base_lo"], errors="coerce").fillna(base_s).to_numpy(dtype="float64")
        hi = pd.to_numeric(df["base_hi"], errors="coerce").fillna(base_s).to_numpy(dtype="float64")
        out["base_center_speed"] = base
        out["hres_center_speed"] = hres
        out["left_width"] = np.maximum(0.03, base - lo)
        out["right_width"] = np.maximum(0.03, hi - base)
        out["width_sum"] = np.maximum(0.03, hi - lo)

    cats = pd.get_dummies(
        df[["level", "season", "hres_dir_sector"]].astype(str),
        columns=["level", "season", "hres_dir_sector"],
        dtype="float64",
    )
    out = pd.concat([out, cats], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is None:
        cols = sorted(out.columns)
    else:
        cols = list(columns)
        out = out.reindex(columns=cols, fill_value=0.0)
    return out[cols], cols


def model(seed: int, loss: str = "squared_error", quantile: float | None = None) -> HistGradientBoostingRegressor:
    kwargs: dict[str, Any] = {
        "loss": loss,
        "learning_rate": 0.045,
        "max_iter": 160,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.05,
        "min_samples_leaf": 35,
        "random_state": int(seed),
    }
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    return HistGradientBoostingRegressor(**kwargs)


def fit_direction_models(train: pd.DataFrame) -> dict[str, Any]:
    X, columns = make_features(train)
    actual_x, actual_y = angle_to_xy(train["actual"].to_numpy(dtype="float64"))
    base_x, base_y = angle_to_xy(train["base_center"].to_numpy(dtype="float64"))
    models = {
        "columns": columns,
        "res_x": model(SEED + 11).fit(X, actual_x - base_x),
        "res_y": model(SEED + 12).fit(X, actual_y - base_y),
        "dir_x": model(SEED + 13).fit(X, actual_x),
        "dir_y": model(SEED + 14).fit(X, actual_y),
    }
    return models


def direction_raw_center(rows: pd.DataFrame, fitted: dict[str, Any], method: str) -> np.ndarray:
    X, _ = make_features(rows, fitted["columns"])
    base_x, base_y = angle_to_xy(rows["base_center"].to_numpy(dtype="float64"))
    if method == "residual_uv":
        x = base_x + fitted["res_x"].predict(X)
        y = base_y + fitted["res_y"].predict(X)
    elif method == "direct_uv":
        x = fitted["dir_x"].predict(X)
        y = fitted["dir_y"].predict(X)
    else:
        raise ValueError(method)
    norm = np.sqrt(x * x + y * y)
    bad = ~np.isfinite(norm) | (norm < 1e-6)
    x[bad] = base_x[bad]
    y[bad] = base_y[bad]
    return xy_to_angle(x, y)


def constrained_direction(rows: pd.DataFrame, raw: np.ndarray, selector: str, thresholds: dict[str, float], weight: float, cap: float, gate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = rows["base_center"].to_numpy(dtype="float64") % 360.0
    delta = signed_circ_diff(raw, base)
    active = RNS.selector_mask(rows, selector, thresholds) & np.isfinite(delta) & (np.abs(delta) <= float(gate))
    offset = np.zeros(len(rows), dtype="float64")
    offset[active] = np.clip(delta[active], -float(cap), float(cap)) * float(weight)
    center = (base + offset) % 360.0
    return center, np.abs(offset), active


def fit_speed_models(train: pd.DataFrame) -> dict[str, Any]:
    X, columns = make_features(train)
    y = train["actual"].to_numpy(dtype="float64")
    mid = train["base_center"].to_numpy(dtype="float64")
    left_need = np.maximum(0.03, mid - y)
    right_need = np.maximum(0.03, y - mid)
    return {
        "columns": columns,
        "left": model(SEED + 21, loss="quantile", quantile=0.90).fit(X, left_need),
        "right": model(SEED + 22, loss="quantile", quantile=0.90).fit(X, right_need),
    }


def learned_speed_interval(rows: pd.DataFrame, fitted: dict[str, Any], selector: str, thresholds: dict[str, float], weight: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, _ = make_features(rows, fitted["columns"])
    mid = rows["base_center"].to_numpy(dtype="float64")
    lo0 = rows["base_lo"].to_numpy(dtype="float64")
    hi0 = rows["base_hi"].to_numpy(dtype="float64")
    left0 = np.maximum(0.03, mid - lo0)
    right0 = np.maximum(0.03, hi0 - mid)
    left_pred = np.maximum(0.03, fitted["left"].predict(X)) * float(scale)
    right_pred = np.maximum(0.03, fitted["right"].predict(X)) * float(scale)
    left = (1.0 - float(weight)) * left0 + float(weight) * left_pred
    right = (1.0 - float(weight)) * right0 + float(weight) * right_pred
    active = RNS.selector_mask(rows, selector, thresholds)
    lo = lo0.copy()
    hi = hi0.copy()
    lo_new = np.maximum(0.0, mid - left)
    hi_new = np.maximum(mid, mid + right)
    lo[active] = lo_new[active]
    hi[active] = hi_new[active]
    move = np.zeros(len(rows), dtype="float64")
    move[active] = np.maximum(np.abs(lo[active] - lo0[active]), np.abs(hi[active] - hi0[active]))
    return lo, hi, move, active


def score_dir(actual: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(actual) & np.isfinite(center)
    if int(good.sum()) < 40:
        return np.nan, np.nan
    score, hw = DAB.cws(actual[good], center[good])
    return float(score), float(hw)


def score_speed(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    good = np.isfinite(actual) & np.isfinite(lo) & np.isfinite(hi)
    if int(good.sum()) < 40:
        return np.nan
    return float(SAB.winkler(actual[good], lo[good], hi[good]))


def evaluate_direction(target: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, val_year: int) -> list[dict[str, Any]]:
    fitted = fit_direction_models(train)
    thresholds = RNS.selector_thresholds(train)
    actual = val["actual"].to_numpy(dtype="float64")
    base = val["base_center"].to_numpy(dtype="float64")
    base_score, base_hw = score_dir(actual, base)
    rows: list[dict[str, Any]] = [
        baseline_row(target, val_year, base_score, base_hw, len(val), "current_model")
    ]
    raw_cache = {method: direction_raw_center(val, fitted, method) for method in DIR_METHODS}
    for method in DIR_METHODS:
        raw = raw_cache[method]
        for selector in DIR_SELECTORS:
            for weight in DIR_WEIGHTS:
                for cap in DIR_CAPS:
                    for gate in DIR_RAW_GATES:
                        center, move, active = constrained_direction(val, raw, selector, thresholds, weight, cap, gate)
                        score, hw = score_dir(actual, center)
                        if not np.isfinite(score):
                            continue
                        regime_base, _ = score_dir(actual[active], base[active])
                        regime_score, _ = score_dir(actual[active], center[active])
                        regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
                        rows.append(
                            candidate_row(
                                target,
                                val_year,
                                f"dirlrn|method={method}|selector={selector}|w={weight:.2f}|cap={cap:.1f}|gate={gate:.1f}",
                                score,
                                hw,
                                base_score,
                                base_hw,
                                move,
                                active,
                                train_selected=int(RNS.selector_mask(train, selector, thresholds).sum()),
                                regime_base=regime_base,
                                regime_score=regime_score,
                                regime_gain=regime_gain,
                            )
                        )
    return rows


def evaluate_speed(target: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, val_year: int) -> list[dict[str, Any]]:
    fitted = fit_speed_models(train)
    thresholds = RNS.selector_thresholds(train)
    actual = val["actual"].to_numpy(dtype="float64")
    lo0 = val["base_lo"].to_numpy(dtype="float64")
    hi0 = val["base_hi"].to_numpy(dtype="float64")
    base_score = score_speed(actual, lo0, hi0)
    base_hw = float(np.nanmedian((hi0 - lo0) / 2.0))
    rows: list[dict[str, Any]] = [
        baseline_row(target, val_year, base_score, base_hw, len(val), "current_model")
    ]
    for selector in SPEED_SELECTORS:
        for weight in SPEED_WEIGHTS:
            for scale in SPEED_SCALES:
                lo, hi, move, active = learned_speed_interval(val, fitted, selector, thresholds, weight, scale)
                score = score_speed(actual, lo, hi)
                if not np.isfinite(score):
                    continue
                regime_base = score_speed(actual[active], lo0[active], hi0[active])
                regime_score = score_speed(actual[active], lo[active], hi[active])
                regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
                rows.append(
                    candidate_row(
                        target,
                        val_year,
                        f"speedlrn|selector={selector}|w={weight:.2f}|scale={scale:.2f}",
                        score,
                        float(np.nanmedian((hi - lo) / 2.0)),
                        base_score,
                        base_hw,
                        move,
                        active,
                        train_selected=int(RNS.selector_mask(train, selector, thresholds).sum()),
                        regime_base=regime_base,
                        regime_score=regime_score,
                        regime_gain=regime_gain,
                    )
                )
    return rows


def baseline_row(target: dict[str, Any], val_year: int, score: float, half_width: float, n: int, candidate: str) -> dict[str, Any]:
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": target["problem"],
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "score": float(score),
        "half_width": float(half_width),
        "baseline_score": float(score),
        "baseline_half_width": float(half_width),
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
        "eval_rows": int(n),
        "scored_values": int(n),
    }


def candidate_row(
    target: dict[str, Any],
    val_year: int,
    candidate: str,
    score: float,
    half_width: float,
    base_score: float,
    base_half_width: float,
    move: np.ndarray,
    active: np.ndarray,
    train_selected: int,
    regime_base: float,
    regime_score: float,
    regime_gain: float,
) -> dict[str, Any]:
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": target["problem"],
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "score": float(score),
        "half_width": float(half_width),
        "baseline_score": float(base_score),
        "baseline_half_width": float(base_half_width),
        "gain": float(base_score - score),
        "regime_score": float(regime_score) if np.isfinite(regime_score) else np.nan,
        "regime_baseline_score": float(regime_base) if np.isfinite(regime_base) else np.nan,
        "regime_gain": float(regime_gain) if np.isfinite(regime_gain) else np.nan,
        "move_mean": float(np.nanmean(move)),
        "move_p90": float(np.nanquantile(move, 0.90)),
        "move_p99": float(np.nanquantile(move, 0.99)),
        "changed_fraction": float(np.mean(np.round(move, 2) > 0.0)),
        "train_selected": int(train_selected),
        "val_selected": int(active.sum()),
        "eval_rows": int(len(move)),
        "scored_values": int(len(move)),
    }


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for target in RNS.TARGETS:
        tid = str(target["target_id"])
        tr = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
        for val_year in VAL_YEARS:
            train = tr[tr["origin_year"] < int(val_year)].reset_index(drop=True)
            val = tr[tr["origin_year"].eq(int(val_year))].reset_index(drop=True)
            if train.empty or val.empty:
                continue
            print(f"[cv] {target['display']} val_year={val_year} train={len(train):,} val={len(val):,}", flush=True)
            if target["problem"] == "dir":
                out.extend(evaluate_direction(target, train, val, val_year))
            else:
                out.extend(evaluate_speed(target, train, val, val_year))
    return pd.DataFrame(out)


def summarize_and_gate(folds: pd.DataFrame, gap_info: dict[str, dict[str, Any]]) -> pd.DataFrame:
    summary = (
        folds.groupby(["target_id", "display", "problem", "region", "group", "horizon", "candidate"], as_index=False)
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
        target = RNS.TARGET_BY_ID[str(row["target_id"])]
        gates_cfg = target["gates"]
        gap = gap_info.get(str(row["display"]), RNS.FALLBACK_GAPS[str(row["display"])])
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
        "row_cache": str(ROW_CACHE),
        "cv_by_fold_csv": str(CV_BY_FOLD_CSV),
        "cv_summary_csv": str(CV_SUMMARY_CSV),
        "decision_csv": str(DECISION_CSV),
        "out_csv": str(OUT_CSV) if OUT_CSV.exists() else None,
        "out_zip": str(OUT_ZIP) if OUT_ZIP.exists() else None,
        "targets": RNS.TARGETS,
        "gate_policy": {
            "val_years": list(VAL_YEARS),
            "direction_model": "HistGradientBoosting learned UV residual/direct-angle models over official anchor rows",
            "speed_model": "HistGradientBoosting quantile width model with q50 locked",
            "selectors": {"direction": list(DIR_SELECTORS), "speed": list(SPEED_SELECTORS)},
            "public_feedback_use": "leaderboard snapshot is used only for target eligibility and safety thresholds",
        },
        "competition_rule_notes": [
            "Uses official phase1 training features/reanalysis labels, current generated base predictions, and inference features only.",
            "No external data or hidden/scoring-server labels are used.",
            "Public leaderboard values are aggregate gates only and are never row-level training labels or features.",
            "Submission output is emitted only if a learned residual candidate clears mean, worst-fold, regime, score, movement, and row-scope gates.",
        ],
        "code_hashes": {
            "builder": sha256(Path(__file__).resolve()),
            "runner": sha256(ROOT / "run_learned_residual_newsignal_v1_e2e.ps1"),
            "regime_builder": sha256(ROOT / "build_regime_newsignal_v1_candidate.py"),
            "base_csv": sha256(BASE_CSV),
        },
    }
    data.update(payload)
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST}", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cleanup_outputs()
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing current-best base CSV: {BASE_CSV}")
    if not ROW_CACHE.exists():
        raise SystemExit(f"Missing row cache from regime branch: {ROW_CACHE}. Run run_regime_newsignal_v1_e2e.ps1 with sample=60 first.")

    rows = pd.read_parquet(ROW_CACHE)
    gap_info = RNS.load_rank_gap()
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
                "reason": "No learned residual candidate cleared strict public-gap, mean-gain, worst-fold, regime, score-ceiling, movement, and coverage gates.",
                "candidates_evaluated": int(len(decisions)),
                "selected": [],
                "top_by_target": top.to_dict("records"),
            },
        )
        return

    # Submission path is intentionally conservative. A passing learned candidate should be rare;
    # keep the implementation explicit if the CV gates ever select one.
    cleanup_outputs()
    write_manifest(
        "blocked_submission_path_not_enabled",
        {
            "reason": "At least one learned candidate passed CV gates, but final inference application is disabled pending manual audit of movement and row scope.",
            "selected": selected,
        },
    )


if __name__ == "__main__":
    main()
