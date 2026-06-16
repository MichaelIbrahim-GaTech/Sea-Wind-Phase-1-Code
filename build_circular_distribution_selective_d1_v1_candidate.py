from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_final_submission as AUD
import build_circular_distribution_direction_v1_candidate as CD
import build_dir_error_width_gridlong_v1_candidate as GL
import build_dir_error_width_newsignal_v1_candidate as DEW
import build_dir_interval_newsignal_v1_candidate as DIW
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_direrrw_ecss14push_v1.csv"
OUT_CSV = WORK / "pred_nsp1cdsel_v1.csv"
OUT_ZIP = WORK / "sub_nsp1cdsel_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_circdist_nsp1cdsel_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_circdist_nsp1cdsel_v1_summary.csv"
DECISION_CSV = WORK / "decision_circdist_nsp1cdsel_v1.csv"
MANIFEST = WORK / "manifest_circdist_nsp1cdsel_v1.json"

TARGET_ID = "dir_ns_pressure_d1"
VAL_YEARS = (2020, 2021)

GROUP_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("speed_level_hour", ("speed_bin", "level", "hour")),
    ("sector_level_hour", ("hres_dir_sector", "level", "hour")),
    ("delta_level_hour", ("delta_bin", "level", "hour")),
    ("level_hour", ("level", "hour")),
)
CENTER_WEIGHTS = (0.25, 0.50)
WIDTH_WEIGHTS = (0.25, 0.50)
WIDTH_QUANTILES = (0.80, 0.90)
MIN_COUNTS = (120, 300)
SELECTORS = ("bias_top", "risk_top", "width_top")
SELECT_FRACTIONS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94)


def to_jsonable(obj: Any) -> Any:
    return GL.to_jsonable(obj)


def remove_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP):
        if path.exists():
            path.unlink()


def selector_score(selector: str, bias: np.ndarray, width_q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.full(len(width_q), np.nan, dtype="float64")
    if selector == "bias_top":
        raw = np.abs(bias)
    elif selector == "width_top":
        raw = width_q
    elif selector == "risk_top":
        raw = width_q + 0.5 * np.abs(bias)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    score[valid] = raw[valid]
    return score


def train_threshold(
    train: pd.DataFrame,
    stats: pd.DataFrame,
    cols: tuple[str, ...],
    min_count: int,
    selector: str,
    select_fraction: float,
) -> tuple[float, int, float]:
    bias, width_q, valid = CD.map_stats(train, stats, cols, min_count)
    score = selector_score(selector, bias, width_q, valid)
    finite = np.isfinite(score)
    if int(finite.sum()) == 0:
        return float("nan"), 0, 0.0
    threshold = float(np.nanquantile(score[finite], 1.0 - float(select_fraction)))
    selected = finite & (score >= threshold)
    return threshold, int(selected.sum()), float(selected.mean()) if len(selected) else 0.0


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
        "score": float(score),
        "baseline_score": float(score),
        "gain": 0.0,
        "regime_gain_min": 0.0,
        "center_move_mean": 0.0,
        "center_move_p90": 0.0,
        "center_move_p99": 0.0,
        "width_move_mean": 0.0,
        "width_move_p90": 0.0,
        "width_move_p99": 0.0,
        "changed_fraction": 0.0,
        "selected_fraction": 0.0,
        "valid_fraction": 1.0,
        "train_selected_fraction": 0.0,
        "threshold": 0.0,
        "train_rows": 0,
        "val_rows": int(len(val)),
        "train_selected": 0,
        "val_selected": int(len(val)),
        "score_values": int(np.isfinite(actual).sum()),
    }


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
    selector: str,
    select_fraction: float,
) -> dict[str, Any]:
    stats = CD.circular_group_stats(train, group_cols, width_quantile)
    threshold, train_selected, train_selected_fraction = train_threshold(
        train, stats, group_cols, min_count, selector, select_fraction
    )
    bias, width_q, valid = CD.map_stats(val, stats, group_cols, min_count)
    score_signal = selector_score(selector, bias, width_q, valid)
    selected = np.isfinite(score_signal) & np.isfinite(threshold) & (score_signal >= threshold)

    actual = val["actual"].to_numpy(dtype="float64")
    base_center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    pred_center = base_center.copy()
    pred_hw = base_hw.copy()
    pred_center[selected] = (base_center[selected] + center_weight * bias[selected]) % 360.0
    if width_weight > 0:
        raw_hw = np.clip(width_q, 5.0, 179.9)
        pred_hw[selected] = np.clip((1.0 - width_weight) * base_hw[selected] + width_weight * raw_hw[selected], 5.0, 179.9)

    baseline_score = DIW.direction_score_var(actual, base_center, base_hw)
    score = DIW.direction_score_var(actual, pred_center, pred_hw)
    center_move = DIW.circ_abs_diff(pred_center, base_center)
    width_move = np.abs(pred_hw - base_hw)
    changed = (center_move > 0.01) | (width_move > 0.01)
    candidate = (
        f"circsel|grp={group_name}|sel={selector}|frac={select_fraction:.2f}|"
        f"cw={center_weight:.2f}|ww={width_weight:.2f}|q={width_quantile:.2f}|minn={int(min_count)}"
    )
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
        "selector": selector,
        "select_fraction": float(select_fraction),
        "center_weight": float(center_weight),
        "width_weight": float(width_weight),
        "width_quantile": float(width_quantile),
        "min_count": int(min_count),
        "score": float(score),
        "baseline_score": float(baseline_score),
        "gain": float(baseline_score - score),
        "regime_gain_min": CD.regime_gain_min(val, pred_center, pred_hw),
        "base_hw_mean": float(np.nanmean(base_hw)),
        "pred_hw_mean": float(np.nanmean(pred_hw)),
        "center_move_mean": float(np.nanmean(center_move)),
        "center_move_p90": float(np.nanquantile(center_move, 0.90)),
        "center_move_p99": float(np.nanquantile(center_move, 0.99)),
        "width_move_mean": float(np.nanmean(width_move)),
        "width_move_p90": float(np.nanquantile(width_move, 0.90)),
        "width_move_p99": float(np.nanquantile(width_move, 0.99)),
        "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "selected_fraction": float(selected.mean()) if len(selected) else 0.0,
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "train_selected_fraction": float(train_selected_fraction),
        "threshold": float(threshold),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_selected": int(train_selected),
        "val_selected": int(selected.sum()),
        "score_values": int(np.isfinite(actual).sum()),
    }


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    target = CD.TARGET_BY_ID[TARGET_ID]
    target_rows = rows[rows["target_id"].eq(TARGET_ID)].reset_index(drop=True)
    if target_rows.empty:
        raise SystemExit(f"No rows found for {TARGET_ID}")
    out: list[dict[str, Any]] = []
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
                            for selector in SELECTORS:
                                for select_fraction in SELECT_FRACTIONS:
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
                                            selector,
                                            select_fraction,
                                        )
                                    )
    return pd.DataFrame(out)


def gates() -> dict[str, float]:
    public_gap = float(CD.PUBLIC_CURRENT_V1.get(TARGET_ID, {}).get("gap_to_best", 0.0))
    return {
        "public_gap": public_gap,
        "min_public_gap": 1.0,
        "min_mean_gain": 6.0,
        "min_worst_gain": 1.5,
        "min_regime_gain": 1.0,
        "min_changed_fraction": 0.03,
        "max_changed_fraction": 0.95,
        "min_selected_fraction": 0.03,
        "max_selected_fraction": 0.95,
        "min_valid_fraction": 0.12,
        "max_center_move_p90": 30.0,
        "max_center_move_p99": 70.0,
        "max_width_move_p90": 50.0,
        "max_width_move_p99": 95.0,
        "min_val_selected": 1200,
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
            selected_fraction=("selected_fraction", "mean"),
            valid_fraction=("valid_fraction", "mean"),
            train_selected_fraction=("train_selected_fraction", "mean"),
            val_selected_min=("val_selected", "min"),
            train_selected_min=("train_selected", "min"),
            fold_count=("val_year", "nunique"),
            val_rows_min=("val_rows", "min"),
        )
        .reset_index(drop=True)
    )
    g = gates()
    public = CD.PUBLIC_CURRENT_V1.get(TARGET_ID, {"ours": None, "top_best": None, "top_best_name": "", "gap_to_best": 0.0})
    gated: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        checks = {
            "not_baseline": str(row["candidate"]) != "current_base_row_cache",
            "public_gap": float(g["public_gap"]) >= float(g["min_public_gap"]),
            "mean_gain": float(row["gain"]) >= float(g["min_mean_gain"]),
            "worst_fold_gain": float(row["gain_min"]) >= float(g["min_worst_gain"]),
            "regime_gain": np.isfinite(float(row["regime_gain_min"])) and float(row["regime_gain_min"]) >= float(g["min_regime_gain"]),
            "changed_fraction": float(g["min_changed_fraction"]) <= float(row["changed_fraction"]) <= float(g["max_changed_fraction"]),
            "selected_fraction": float(g["min_selected_fraction"]) <= float(row["selected_fraction"]) <= float(g["max_selected_fraction"]),
            "valid_fraction": float(row["valid_fraction"]) >= float(g["min_valid_fraction"]),
            "center_move_p90": float(row["center_move_p90"]) <= float(g["max_center_move_p90"]),
            "center_move_p99": float(row["center_move_p99"]) <= float(g["max_center_move_p99"]),
            "width_move_p90": float(row["width_move_p90"]) <= float(g["max_width_move_p90"]),
            "width_move_p99": float(row["width_move_p99"]) <= float(g["max_width_move_p99"]),
            "val_selected": int(row["val_selected_min"]) >= int(g["min_val_selected"]),
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
        ["gate_passed_cv", "score_max", "gain", "gain_min", "regime_gain_min", "center_move_p90"],
        ascending=[False, True, False, False, False, True],
        kind="mergesort",
    )


def parse_candidate(candidate: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for part in candidate.split("|")[1:]:
        key, value = part.split("=", 1)
        params[key] = value
    return {
        "group_name": str(params["grp"]),
        "selector": str(params["sel"]),
        "select_fraction": float(params["frac"]),
        "center_weight": float(params["cw"]),
        "width_weight": float(params["ww"]),
        "width_quantile": float(params["q"]),
        "min_count": int(params["minn"]),
    }


def select_candidate(decisions: pd.DataFrame) -> dict[str, Any] | None:
    passed = decisions[decisions["gate_passed_cv"].astype(bool)].copy()
    if passed.empty:
        return None
    selected = passed.sort_values(
        ["score_max", "gain", "gain_min", "regime_gain_min", "center_move_p90"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).head(1)
    return selected.iloc[0].to_dict()


def inference_add_bins(inf: pd.DataFrame) -> pd.DataFrame:
    out = inf.copy()
    if "actual" not in out.columns:
        out["actual"] = pd.to_numeric(out["base_center"], errors="coerce").to_numpy(dtype="float64") % 360.0
    return CD.add_bins(out)


def apply_selected_to_submission(df: pd.DataFrame, rows: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    target = CD.TARGET_BY_ID[TARGET_ID]
    params = parse_candidate(str(selected["candidate"]))
    group_cols = dict(GROUP_SPECS)[params["group_name"]]
    train = rows[rows["target_id"].eq(TARGET_ID)].reset_index(drop=True)
    stats = CD.circular_group_stats(train, group_cols, params["width_quantile"])
    threshold, train_selected, train_selected_fraction = train_threshold(
        train,
        stats,
        group_cols,
        params["min_count"],
        params["selector"],
        params["select_fraction"],
    )

    inf = DEW.inference_rows_for_target(df, target)
    inf = inference_add_bins(inf)
    bias, width_q, valid = CD.map_stats(inf, stats, group_cols, params["min_count"])
    score_signal = selector_score(params["selector"], bias, width_q, valid)
    selected_mask = np.isfinite(score_signal) & np.isfinite(threshold) & (score_signal >= threshold)

    idx = inf["_submission_index"].to_numpy(dtype="int64")
    base_center = inf["base_center"].to_numpy(dtype="float64")
    base_hw = inf["base_hw"].to_numpy(dtype="float64")
    pred_center = base_center.copy()
    pred_hw = base_hw.copy()
    pred_center[selected_mask] = (base_center[selected_mask] + params["center_weight"] * bias[selected_mask]) % 360.0
    if params["width_weight"] > 0:
        raw_hw = np.clip(width_q, 5.0, 179.9)
        pred_hw[selected_mask] = np.clip(
            (1.0 - params["width_weight"]) * base_hw[selected_mask] + params["width_weight"] * raw_hw[selected_mask],
            5.0,
            179.9,
        )

    center_move = DIW.circ_abs_diff(pred_center, base_center)
    width_move = np.abs(pred_hw - base_hw)
    changed = (center_move > 0.01) | (width_move > 0.01)
    g = gates()
    checks = {
        "selected_fraction": float(g["min_selected_fraction"]) <= float(selected_mask.mean()) <= float(g["max_selected_fraction"]),
        "changed_fraction": float(g["min_changed_fraction"]) <= float(changed.mean()) <= float(g["max_changed_fraction"]),
        "valid_fraction": float(valid.mean()) >= float(g["min_valid_fraction"]),
        "center_move_p90": float(np.nanquantile(center_move, 0.90)) <= float(g["max_center_move_p90"]),
        "center_move_p99": float(np.nanquantile(center_move, 0.99)) <= float(g["max_center_move_p99"]),
        "width_move_p90": float(np.nanquantile(width_move, 0.90)) <= float(g["max_width_move_p90"]),
        "width_move_p99": float(np.nanquantile(width_move, 0.99)) <= float(g["max_width_move_p99"]),
    }
    audit = {
        "target_id": TARGET_ID,
        "display": str(target["display"]),
        "candidate": str(selected["candidate"]),
        "rows_in_scope": int(len(idx)),
        "rows_selected": int(selected_mask.sum()),
        "rows_changed": int(changed.sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "selected_fraction": float(selected_mask.mean()) if len(selected_mask) else 0.0,
        "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "train_selected": int(train_selected),
        "train_selected_fraction": float(train_selected_fraction),
        "threshold": float(threshold),
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


def write_manifest(status: str, payload: dict[str, Any]) -> None:
    data = {
        "status": status,
        "mode": "circular_distribution_selective_d1_v1",
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": GL.sha256(BASE_CSV),
        "row_source": {
            "gridlong_cache": str(CD.GRIDLONG_CACHE),
            "feature_cache": str(CD.FEATURE_CACHE),
            "target_id": TARGET_ID,
        },
        "cv_by_fold_csv": str(CV_BY_FOLD_CSV),
        "cv_summary_csv": str(CV_SUMMARY_CSV),
        "decision_csv": str(DECISION_CSV),
        "out_csv": str(OUT_CSV) if OUT_CSV.exists() else None,
        "out_zip": str(OUT_ZIP) if OUT_ZIP.exists() else None,
        "gate_policy": {
            "val_years": VAL_YEARS,
            "group_specs": GROUP_SPECS,
            "center_weights": CENTER_WEIGHTS,
            "width_weights": WIDTH_WEIGHTS,
            "width_quantiles": WIDTH_QUANTILES,
            "min_counts": MIN_COUNTS,
            "selectors": SELECTORS,
            "select_fractions": SELECT_FRACTIONS,
            "gates": gates(),
            "public_feedback_use": "aggregate target-level priority and rollback gates only",
        },
        "competition_rule_notes": [
            "Uses only official phase1 training features/reanalysis labels, generated base predictions, and official inference feature files.",
            "No external data, hidden labels, row-level scoring-server labels, or public leaderboard row labels are used.",
            "Public leaderboard values are used only as aggregate target-priority and safety gates.",
            "The selector threshold is learned from training-fold rows only, then applied to validation/inference rows.",
            "The branch is fail-closed: no submission is emitted unless CV and inference gates pass.",
        ],
        "code_hashes": {
            "builder": GL.sha256(Path(__file__).resolve()),
            "circular_distribution_direction_v1": GL.sha256(ROOT / "build_circular_distribution_direction_v1_candidate.py"),
            "base_csv": GL.sha256(BASE_CSV),
        },
    }
    data.update(payload)
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST}", flush=True)


def append_decision_log(status: str, decisions: pd.DataFrame, selected: dict[str, Any] | None) -> None:
    path = ROOT / "submission_decisions.md"
    top = decisions[
        [
            "target_id",
            "candidate",
            "score",
            "score_max",
            "gain",
            "gain_min",
            "regime_gain_min",
            "changed_fraction",
            "selected_fraction",
            "gate_passed_cv",
            "reject_reasons",
        ]
    ].head(8)
    lines = [
        "",
        "## Selective Circular Distribution D1 Branch",
        "",
        f"- Status: `{status}`.",
        f"- Base: `{BASE_CSV}`.",
        f"- Target: `{TARGET_ID}`.",
        f"- Decision CSV: `{DECISION_CSV}`.",
        f"- Submission zip: `{OUT_ZIP if OUT_ZIP.exists() else 'none written'}`.",
        f"- Selected candidate: `{selected.get('candidate') if selected else 'none'}`.",
        "- Rule note: official data and generated predictions only; public leaderboard aggregates used only as target-priority and rollback gates.",
        "",
        "Top gated rows:",
        "",
        "```csv",
        top.to_csv(index=False).strip(),
        "```",
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_submission(rows: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any] | None:
    print(f"[submission] reading {BASE_CSV}", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)
    audit = apply_selected_to_submission(df, rows, selected)
    if not audit.get("inference_gate_passed", False):
        remove_outputs()
        write_manifest(
            "inference_gate_blocked_no_submission",
            {
                "selected_candidate": selected,
                "inference_audit": audit,
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
    return {
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
        "inference_audit": audit,
    }


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing current best base CSV: {BASE_CSV}. Run .\\run_direrrw_ecss14push_v1_e2e.ps1 first.")
    remove_outputs()
    rows = CD.read_rows()
    rows = rows[rows["target_id"].eq(TARGET_ID)].reset_index(drop=True)
    print(f"[rows] loaded {len(rows):,} rows for {TARGET_ID}", flush=True)
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
                "selected_fraction",
                "gate_passed_cv",
                "reject_reasons",
            ]
        ]
        .head(18)
        .to_string(index=False),
        flush=True,
    )

    selected = select_candidate(decisions)
    if selected is None:
        remove_outputs()
        write_manifest(
            "blocked_no_submission",
            {
                "candidates_evaluated": int(len(decisions)),
                "candidates_passed_cv": 0,
                "top_candidates": decisions.head(24).to_dict("records"),
            },
        )
        append_decision_log("blocked_no_submission", decisions, None)
        return

    output = write_submission(rows, selected)
    if output is None:
        append_decision_log("inference_gate_blocked_no_submission", decisions, selected)
        return
    write_manifest(
        "submission_written_after_selective_circular_d1_gate",
        {
            "candidates_evaluated": int(len(decisions)),
            "candidates_passed_cv": int(decisions["gate_passed_cv"].astype(bool).sum()),
            "selected_candidate": selected,
            "output": output,
        },
    )
    append_decision_log("submission_written_after_selective_circular_d1_gate", decisions, selected)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
