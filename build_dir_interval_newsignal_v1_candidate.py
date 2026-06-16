from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_regime_newsignal_v1_candidate as RNS
import direction_anchor_backtest as DAB
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_ns_sfc14_rg90.csv"
ROW_CACHE = WORK / "regime_newsignal_v1_rows_s180.parquet"
OUT_CSV = WORK / "pred_dir_interval_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_dirint_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_dir_interval_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_dir_interval_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_dir_interval_newsignal_v1.csv"
MANIFEST = WORK / "manifest_dir_interval_newsignal_v1.json"

VAL_YEARS = (2020, 2021)
WIDTH_GRID = np.array(
    list(np.arange(15.0, 70.1, 2.5))
    + list(np.arange(75.0, 160.1, 5.0))
    + [165.0, 170.0, 175.0, 179.9],
    dtype="float64",
)
SCALE_GRID = (0.60, 0.70, 0.80, 0.90, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)
GROUPINGS = ("global", "level", "hour", "level_hour")
SHRINK_VALUES = (0.0, 4000.0, 16000.0)
MIN_GROUP_N = 400

TARGET_IDS = {
    "dir_ns_pressure_d1",
    "dir_ecs_pressure_d1",
    "dir_ecs_pressure_d14",
    "dir_ns_surface_d1",
}
TARGETS = tuple(t for t in RNS.DIRECTION_TARGETS if str(t["target_id"]) in TARGET_IDS)
TARGET_BY_ID = {str(t["target_id"]): t for t in TARGETS}

# Width-only moves are safer than center moves, but this branch still blocks
# broad moves unless they are strongly and consistently supported by CV.
WIDTH_GATE_OVERRIDES: dict[str, dict[str, float]] = {
    "dir_ns_pressure_d1": {
        "max_move_p90": 65.0,
        "max_move_p99": 95.0,
        "max_changed_fraction": 1.00,
    },
    "dir_ecs_pressure_d1": {
        "max_move_p90": 65.0,
        "max_move_p99": 95.0,
        "max_changed_fraction": 1.00,
    },
    "dir_ecs_pressure_d14": {
        "max_move_p90": 90.0,
        "max_move_p99": 120.0,
        "max_changed_fraction": 1.00,
    },
    "dir_ns_surface_d1": {
        "max_move_p90": 65.0,
        "max_move_p99": 95.0,
        "max_changed_fraction": 1.00,
    },
}


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
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except TypeError:
        pass
    return obj


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP):
        if path.exists():
            path.unlink()


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def direction_score_var(actual: np.ndarray, center: np.ndarray, half_width: np.ndarray | float) -> float:
    y = np.asarray(actual, dtype="float64") % 360.0
    c = np.asarray(center, dtype="float64") % 360.0
    hw = np.asarray(half_width, dtype="float64")
    if hw.ndim == 0:
        hw = np.full(len(y), float(hw), dtype="float64")
    ok = np.isfinite(y) & np.isfinite(c) & np.isfinite(hw)
    if int(ok.sum()) < 40:
        return float("nan")
    y = y[ok]
    c = c[ok]
    hw = np.clip(hw[ok], 5.0, 179.9)
    lo = (c - hw) % 360.0
    hi = (c + hw) % 360.0
    width = (hi - lo) % 360.0
    inside = ((y - lo) % 360.0) <= width
    miss = np.minimum(circ_abs_diff(y, lo), circ_abs_diff(y, hi))
    return float(np.mean(width + 20.0 * miss * (~inside)))


def base_width_lookup() -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for region in ("north_sea", "east_china_sea"):
        bundle = DAB.load_dir_bundle(region)
        for level, calib in bundle["calibration"].items():
            for horizon, info in calib.items():
                out[(region, str(level), int(horizon))] = float(info["half_width"])
    return out


def attach_base_hw(rows: pd.DataFrame) -> pd.DataFrame:
    width = base_width_lookup()
    out = rows.copy()
    out["base_hw"] = [
        width[(str(region), str(level), int(horizon))]
        for region, level, horizon in zip(out["region"], out["level"], out["horizon"])
    ]
    return out


def group_cols(grouping: str) -> list[str]:
    if grouping == "level":
        return ["level"]
    if grouping == "hour":
        return ["hour"]
    if grouping == "level_hour":
        return ["level", "hour"]
    return []


def group_key(df: pd.DataFrame, grouping: str) -> pd.Series:
    cols = group_cols(grouping)
    if not cols:
        return pd.Series(["__global__"] * len(df), index=df.index)
    return df[cols].astype(str).agg("|".join, axis=1)


def best_width(actual: np.ndarray, center: np.ndarray, grid: np.ndarray = WIDTH_GRID) -> tuple[float, float]:
    best_hw = float(grid[0])
    best_score = float("inf")
    for hw in grid:
        score = direction_score_var(actual, center, float(hw))
        if np.isfinite(score) and score < best_score:
            best_hw = float(hw)
            best_score = float(score)
    return best_hw, best_score


def fit_width_map(train: pd.DataFrame, grouping: str, shrink: float) -> dict[str, Any]:
    y = train["actual"].to_numpy(dtype="float64")
    c = train["base_center"].to_numpy(dtype="float64")
    global_hw, global_score = best_width(y, c)
    cols = group_cols(grouping)
    width_map: dict[str, float] = {}
    group_stats: list[dict[str, Any]] = []
    if cols:
        for key, sub in train.groupby(cols, sort=False, dropna=False):
            if len(sub) < MIN_GROUP_N:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_str = "|".join(map(str, key_tuple))
            raw_hw, raw_score = best_width(sub["actual"].to_numpy(dtype="float64"), sub["base_center"].to_numpy(dtype="float64"))
            weight = len(sub) / (len(sub) + float(shrink))
            fitted_hw = weight * raw_hw + (1.0 - weight) * global_hw
            width_map[key_str] = float(fitted_hw)
            group_stats.append(
                {
                    "key": key_str,
                    "n": int(len(sub)),
                    "raw_hw": float(raw_hw),
                    "fitted_hw": float(fitted_hw),
                    "score": float(raw_score),
                }
            )
    return {
        "kind": "map",
        "grouping": grouping,
        "shrink": float(shrink),
        "global_hw": float(global_hw),
        "global_score": float(global_score),
        "width_map": width_map,
        "group_stats": group_stats,
    }


def predict_map_width(rows: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    grouping = str(fit["grouping"])
    global_hw = float(fit["global_hw"])
    if grouping == "global":
        return np.full(len(rows), global_hw, dtype="float64")
    width_map = dict(fit.get("width_map", {}))
    return group_key(rows, grouping).map(width_map).fillna(global_hw).to_numpy(dtype="float64")


def fit_policy(train: pd.DataFrame, candidate: str) -> dict[str, Any]:
    parts = dict(item.split("=", 1) for item in candidate.split("|")[1:])
    if candidate.startswith("hwmap|"):
        return fit_width_map(train, str(parts["group"]), float(parts["shrink"]))
    if candidate.startswith("hwscale|"):
        return {"kind": "scale", "scale": float(parts["scale"])}
    raise ValueError(f"unknown candidate: {candidate}")


def predict_policy_width(rows: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    if fit["kind"] == "map":
        return predict_map_width(rows, fit)
    if fit["kind"] == "scale":
        return np.clip(rows["base_hw"].to_numpy(dtype="float64") * float(fit["scale"]), 5.0, 179.9)
    raise ValueError(str(fit["kind"]))


def baseline_row(target: dict[str, Any], val: pd.DataFrame, val_year: int) -> dict[str, Any]:
    actual = val["actual"].to_numpy(dtype="float64")
    center = val["base_center"].to_numpy(dtype="float64")
    hw = val["base_hw"].to_numpy(dtype="float64")
    score = direction_score_var(actual, center, hw)
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "dir",
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": "current_model_native_width",
        "score": float(score),
        "half_width": float(np.nanmedian(hw)),
        "baseline_score": float(score),
        "baseline_half_width": float(np.nanmedian(hw)),
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
        "scored_values": int(np.isfinite(actual).sum()),
    }


def candidate_row(target: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, val_year: int, candidate: str) -> dict[str, Any] | None:
    fit = fit_policy(train, candidate)
    actual = val["actual"].to_numpy(dtype="float64")
    center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    pred_hw = predict_policy_width(val, fit)
    score = direction_score_var(actual, center, pred_hw)
    baseline_score = direction_score_var(actual, center, base_hw)
    if not (np.isfinite(score) and np.isfinite(baseline_score)):
        return None
    move = np.abs(pred_hw - base_hw)
    active = np.round(move, 2) > 0.0
    regime_base = direction_score_var(actual[active], center[active], base_hw[active])
    regime_score = direction_score_var(actual[active], center[active], pred_hw[active])
    regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "dir",
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": candidate,
        "score": float(score),
        "half_width": float(np.nanmedian(pred_hw)),
        "baseline_score": float(baseline_score),
        "baseline_half_width": float(np.nanmedian(base_hw)),
        "gain": float(baseline_score - score),
        "regime_score": float(regime_score) if np.isfinite(regime_score) else np.nan,
        "regime_baseline_score": float(regime_base) if np.isfinite(regime_base) else np.nan,
        "regime_gain": float(regime_gain) if np.isfinite(regime_gain) else np.nan,
        "move_mean": float(np.nanmean(move)),
        "move_p90": float(np.nanquantile(move, 0.90)),
        "move_p99": float(np.nanquantile(move, 0.99)),
        "changed_fraction": float(np.mean(active)),
        "train_selected": int(len(train)),
        "val_selected": int(active.sum()),
        "eval_rows": int(len(val)),
        "scored_values": int(np.isfinite(actual).sum()),
        "fit_global_hw": float(fit.get("global_hw", np.nan)),
        "fit_scale": float(fit.get("scale", np.nan)),
    }


def candidate_names() -> list[str]:
    out: list[str] = []
    for grouping in GROUPINGS:
        for shrink in SHRINK_VALUES:
            if grouping == "global" and shrink != 0.0:
                continue
            out.append(f"hwmap|group={grouping}|shrink={shrink:.0f}")
    for scale in SCALE_GRID:
        out.append(f"hwscale|scale={scale:.2f}")
    return out


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    rows = attach_base_hw(rows[rows["target_id"].isin(TARGET_IDS)].copy())
    out: list[dict[str, Any]] = []
    names = candidate_names()
    for target in TARGETS:
        tid = str(target["target_id"])
        target_rows = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
        for val_year in VAL_YEARS:
            train = target_rows[target_rows["origin_year"] < int(val_year)].reset_index(drop=True)
            val = target_rows[target_rows["origin_year"].eq(int(val_year))].reset_index(drop=True)
            if train.empty or val.empty:
                continue
            print(f"[cv] {target['display']} val_year={val_year} train={len(train):,} val={len(val):,}", flush=True)
            out.append(baseline_row(target, val, val_year))
            for name in names:
                rec = candidate_row(target, train, val, val_year, name)
                if rec is not None:
                    out.append(rec)
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
            baseline_half_width=("baseline_half_width", "mean"),
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
            fit_global_hw=("fit_global_hw", "mean"),
            fit_scale=("fit_scale", "mean"),
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        target = TARGET_BY_ID[str(row["target_id"])]
        gates_cfg = dict(target["gates"])
        gates_cfg.update(WIDTH_GATE_OVERRIDES.get(str(row["target_id"]), {}))
        gap = gap_info.get(str(row["display"]), RNS.FALLBACK_GAPS[str(row["display"])])
        gates = {
            "not_baseline": str(row["candidate"]) != "current_model_native_width",
            "public_gap": float(gap["gap_to_best"]) >= float(gates_cfg["min_public_gap"]),
            "mean_gain": float(row["gain"]) >= float(gates_cfg["min_mean_gain"]),
            "worst_gain": float(row["gain_min"]) >= float(gates_cfg["min_worst_gain"]),
            "regime_worst_gain": np.isfinite(float(row["regime_gain_min"])) and float(row["regime_gain_min"]) >= float(gates_cfg["min_regime_gain"]),
            "score_ceiling": float(row["score_max"]) <= float(gates_cfg["max_score_max"]),
            "cv_width_move_p90": float(row["move_p90"]) <= float(gates_cfg["max_move_p90"]),
            "cv_width_move_p99": float(row["move_p99"]) <= float(gates_cfg.get("max_move_p99", 9999.0)),
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


def submission_target_mask(df: pd.DataFrame, target: dict[str, Any]) -> pd.Series:
    return (
        df["type"].eq("grid")
        & df["region"].eq(str(target["region"]))
        & df["horizon"].astype(int).eq(int(target["horizon"]))
        & df["level"].astype(str).isin(tuple(target["levels"]))
    )


def apply_policy_to_submission(df: pd.DataFrame, target: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    mask = submission_target_mask(df, target)
    idx = df.index[mask]
    sub = df.loc[idx, ["level", "hour", "dir_05", "dir_50", "dir_95"]].copy()
    sub["base_hw"] = ((pd.to_numeric(sub["dir_95"], errors="coerce") - pd.to_numeric(sub["dir_05"], errors="coerce")) % 360.0) / 2.0
    if fit["kind"] == "map":
        pred_hw = predict_map_width(sub.rename(columns={"dir_50": "base_center"}), fit)
    else:
        pred_hw = np.clip(sub["base_hw"].to_numpy(dtype="float64") * float(fit["scale"]), 5.0, 179.9)
    center = pd.to_numeric(sub["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
    old_lo = pd.to_numeric(sub["dir_05"], errors="coerce").to_numpy(dtype="float64") % 360.0
    old_hi = pd.to_numeric(sub["dir_95"], errors="coerce").to_numpy(dtype="float64") % 360.0
    new_lo = (center - pred_hw) % 360.0
    new_hi = (center + pred_hw) % 360.0
    changed = (np.round(circ_abs_diff(new_lo, old_lo), 2) > 0.0) | (np.round(circ_abs_diff(new_hi, old_hi), 2) > 0.0)
    df.loc[idx, "dir_05"] = new_lo
    df.loc[idx, "dir_95"] = new_hi
    return {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "rows_in_scope": int(len(idx)),
        "rows_changed": int(changed.sum()),
        "new_half_width_median": float(np.nanmedian(pred_hw)),
        "new_half_width_p10": float(np.nanquantile(pred_hw, 0.10)),
        "new_half_width_p90": float(np.nanquantile(pred_hw, 0.90)),
        "width_move_p90": float(np.nanquantile(np.abs(pred_hw - sub["base_hw"].to_numpy(dtype="float64")), 0.90)),
    }


def write_submission(selected: list[dict[str, Any]], all_rows: pd.DataFrame) -> dict[str, Any]:
    df = pd.read_csv(BASE_CSV, low_memory=False)
    patches: list[dict[str, Any]] = []
    for row in selected:
        target = TARGET_BY_ID[str(row["target_id"])]
        train = attach_base_hw(all_rows[all_rows["target_id"].eq(str(row["target_id"]))].reset_index(drop=True))
        fit = fit_policy(train, str(row["candidate"]))
        patches.append(apply_policy_to_submission(df, target, fit))
    final = E2E.validate_final(df)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise RuntimeError(f"zip validation failed names={names} bad={bad}")
    return {
        "csv": str(OUT_CSV),
        "zip": str(OUT_ZIP),
        "zip_size": int(OUT_ZIP.stat().st_size),
        "predictions_csv_size": int(info.file_size),
        "csv_sha256": sha256(OUT_CSV),
        "zip_sha256": sha256(OUT_ZIP),
        "internal_names": names,
        "testzip": bad,
        "patches": patches,
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
        "targets": TARGETS,
        "gate_policy": {
            "val_years": list(VAL_YEARS),
            "width_grid": [float(x) for x in WIDTH_GRID],
            "scale_grid": [float(x) for x in SCALE_GRID],
            "groupings": list(GROUPINGS),
            "shrink_values": [float(x) for x in SHRINK_VALUES],
            "target_gates": {
                str(t["target_id"]): {**t["gates"], **WIDTH_GATE_OVERRIDES.get(str(t["target_id"]), {})}
                for t in TARGETS
            },
            "public_feedback_use": "leaderboard snapshot is used only for target eligibility and safety thresholds",
        },
        "competition_rule_notes": [
            "Uses official phase1 training features/reanalysis labels, current generated base predictions, and model calibration artifacts only.",
            "No external data or hidden/scoring-server labels are used.",
            "Public leaderboard values are aggregate gates only and are never row-level training labels or features.",
            "Any emitted submission changes only direction interval widths dir_05/dir_95 around existing dir_50 centers.",
        ],
        "code_hashes": {
            "builder": sha256(Path(__file__).resolve()),
            "runner": sha256(ROOT / "run_dir_interval_newsignal_v1_e2e.ps1"),
            "regime_builder": sha256(ROOT / "build_regime_newsignal_v1_candidate.py"),
            "direction_anchor_backtest": sha256(ROOT / "direction_anchor_backtest.py"),
            "base_csv": sha256(BASE_CSV),
        },
    }
    data.update(payload)
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST}", flush=True)


def append_decision_log(status: str, decisions: pd.DataFrame, selected: list[dict[str, Any]]) -> None:
    top = decisions.groupby("display", sort=False).head(1)
    lines = [
        "",
        "## Branch Result - Direction Interval New Signal V1",
        "",
        f"- Runner: `run_dir_interval_newsignal_v1_e2e.ps1`.",
        f"- Builder: `build_dir_interval_newsignal_v1_candidate.py`.",
        f"- Manifest: `{MANIFEST}`.",
        f"- Decisions: `{DECISION_CSV}`.",
        f"- Fold rows: `{CV_BY_FOLD_CSV}`.",
        f"- Submission zip: `{OUT_ZIP if OUT_ZIP.exists() else 'none written'}`.",
        f"- Status: `{status}`.",
        f"- Candidates evaluated: `{len(decisions)}`.",
        f"- CV-passing candidates: `{len(selected)}`.",
        "- Method: width-only circular Winkler calibration using native model half-widths as baseline; centers remain locked.",
        "- Competition rules: official training/reanalysis/model artifacts only; no external data or hidden labels; public values are aggregate fail-closed gates.",
        "- Best candidates by target:",
    ]
    for row in top.to_dict("records"):
        lines.append(
            "  - `{display}`: `{candidate}`; mean gain `{gain:.4f}`, worst-fold gain `{gain_min:.4f}`, "
            "regime worst gain `{regime_gain_min:.4f}`, score max `{score_max:.4f}`, half-width `{half_width:.2f}`; "
            "failed `{reject}`.".format(
                display=row["display"],
                candidate=row["candidate"],
                gain=float(row["gain"]),
                gain_min=float(row["gain_min"]),
                regime_gain_min=float(row["regime_gain_min"]) if pd.notna(row["regime_gain_min"]) else float("nan"),
                score_max=float(row["score_max"]),
                half_width=float(row["half_width"]),
                reject=str(row["reject_reasons"]) or "none",
            )
        )
    if selected:
        lines.append("- Decision: a gated interval-width candidate was emitted for manual/public scoring.")
    else:
        lines.append("- Decision: no submission was emitted; current best remains `runs/v6_pressure_speed/sub_nssfc14rg90.zip`.")
    with (ROOT / "submission_decisions.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cleanup_outputs()
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing current-best base CSV: {BASE_CSV}")
    if not ROW_CACHE.exists():
        raise SystemExit(f"Missing row cache: {ROW_CACHE}. Run run_lgbm_base_residual_newsignal_v1_e2e.ps1 first.")

    rows = pd.read_parquet(ROW_CACHE)
    rows = rows[rows["target_id"].isin(TARGET_IDS)].reset_index(drop=True)
    gap_info = RNS.load_rank_gap()
    folds = run_cv(rows)
    folds.to_csv(CV_BY_FOLD_CSV, index=False)
    decisions = summarize_and_gate(folds, gap_info)
    decisions.to_csv(CV_SUMMARY_CSV, index=False)
    decisions.to_csv(DECISION_CSV, index=False)
    selected = select_candidates(decisions)

    if not selected:
        top = decisions.groupby("display", sort=False).head(5)
        status = "blocked_no_submission"
        write_manifest(
            status,
            {
                "reason": "No direction interval-width candidate cleared strict public-gap, mean-gain, worst-fold, regime, score-ceiling, width-movement, and coverage gates.",
                "candidates_evaluated": int(len(decisions)),
                "selected": [],
                "top_by_target": top.to_dict("records"),
            },
        )
        append_decision_log(status, decisions, selected)
        return

    output = write_submission(selected, rows)
    status = "submission_written"
    write_manifest(
        status,
        {
            "reason": "At least one direction interval-width policy cleared strict CV gates and was applied with centers locked.",
            "candidates_evaluated": int(len(decisions)),
            "selected": selected,
            "submission": output,
        },
    )
    append_decision_log(status, decisions, selected)


if __name__ == "__main__":
    main()
