from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import build_dir_interval_newsignal_v1_candidate as DIW
import build_feature_rich_newsignal_v1_candidate as FR
import build_regime_newsignal_v1_candidate as RNS
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

BASE_CSV = WORK / "pred_ns_sfc14_rg90.csv"
ROW_CACHE = WORK / "feature_rich_newsignal_v1_rows_s180.parquet"
OUT_CSV = WORK / "pred_dir_error_width_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_direrrw_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_dir_error_width_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_dir_error_width_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_dir_error_width_newsignal_v1.csv"
MANIFEST = WORK / "manifest_dir_error_width_newsignal_v1.json"

VAL_YEARS = (2020, 2021)
ALPHAS = (0.70, 0.80, 0.90, 0.95)
WEIGHTS = (0.25, 0.50, 0.75, 1.00)
SCALES = (0.70, 0.85, 1.00, 1.15, 1.30)
TARGETS = DIW.TARGETS
TARGET_IDS = DIW.TARGET_IDS
TARGET_BY_ID = DIW.TARGET_BY_ID


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


def season_from_month(month: int) -> str:
    return RNS.season_from_month(int(month))


def hres_lead(horizon: int) -> int:
    return int(horizon) if int(horizon) in (1, 7) else 10


def make_features(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    base, base_cols = FR.make_feature_rich_features(df, None)
    extra = pd.DataFrame(index=df.index)
    hw = pd.to_numeric(df["base_hw"], errors="coerce").fillna(30.0).to_numpy(dtype="float64")
    extra["base_hw"] = hw
    extra["base_hw_log1p"] = np.log1p(np.maximum(0.0, hw))
    extra["base_hw_inv"] = 1.0 / np.maximum(1.0, hw)
    out = pd.concat([base, extra], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_cols = list(base_cols) + list(extra.columns)
    if columns is None:
        return out[all_cols], all_cols
    cols = list(columns)
    out = out.reindex(columns=cols, fill_value=0.0)
    return out[cols], cols


def model(seed: int, alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=7,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=6.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def fit_error_model(train: pd.DataFrame, alpha: float, seed: int) -> dict[str, Any]:
    X, cols = make_features(train)
    err = DIW.circ_abs_diff(train["actual"].to_numpy(dtype="float64"), train["base_center"].to_numpy(dtype="float64"))
    err = np.clip(err, 1.0, 179.9)
    fitted = model(seed, alpha).fit(X, err)
    return {"alpha": float(alpha), "columns": cols, "model": fitted}


def predict_error_width(rows: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    X, _ = make_features(rows, fitted["columns"])
    return np.clip(np.asarray(fitted["model"].predict(X), dtype="float64"), 5.0, 179.9)


def baseline_row(target: dict[str, Any], val: pd.DataFrame, val_year: int) -> dict[str, Any]:
    actual = val["actual"].to_numpy(dtype="float64")
    center = val["base_center"].to_numpy(dtype="float64")
    hw = val["base_hw"].to_numpy(dtype="float64")
    score = DIW.direction_score_var(actual, center, hw)
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


def candidate_row(
    target: dict[str, Any],
    train_n: int,
    val: pd.DataFrame,
    val_year: int,
    raw_width: np.ndarray,
    alpha: float,
    weight: float,
    scale: float,
) -> dict[str, Any] | None:
    actual = val["actual"].to_numpy(dtype="float64")
    center = val["base_center"].to_numpy(dtype="float64")
    base_hw = val["base_hw"].to_numpy(dtype="float64")
    learned = np.clip(raw_width * float(scale), 5.0, 179.9)
    pred_hw = np.clip((1.0 - float(weight)) * base_hw + float(weight) * learned, 5.0, 179.9)
    score = DIW.direction_score_var(actual, center, pred_hw)
    baseline_score = DIW.direction_score_var(actual, center, base_hw)
    if not (np.isfinite(score) and np.isfinite(baseline_score)):
        return None
    move = np.abs(pred_hw - base_hw)
    active = np.round(move, 2) > 0.0
    regime_base = DIW.direction_score_var(actual[active], center[active], base_hw[active])
    regime_score = DIW.direction_score_var(actual[active], center[active], pred_hw[active])
    regime_gain = regime_base - regime_score if np.isfinite(regime_base) and np.isfinite(regime_score) else np.nan
    return {
        "target_id": target["target_id"],
        "display": target["display"],
        "problem": "dir",
        "region": target["region"],
        "group": target["group"],
        "horizon": int(target["horizon"]),
        "val_year": int(val_year),
        "candidate": f"errq|alpha={alpha:.2f}|w={weight:.2f}|scale={scale:.2f}",
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
        "train_selected": int(train_n),
        "val_selected": int(active.sum()),
        "eval_rows": int(len(val)),
        "scored_values": int(np.isfinite(actual).sum()),
        "pred_hw_p10": float(np.nanquantile(pred_hw, 0.10)),
        "pred_hw_p90": float(np.nanquantile(pred_hw, 0.90)),
    }


def evaluate_target_fold(target: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, val_year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [baseline_row(target, val, val_year)]
    for a_idx, alpha in enumerate(ALPHAS):
        print(f"  [fit] alpha={alpha:.2f}", flush=True)
        fitted = fit_error_model(train, alpha, seed=20260614 + int(val_year) * 100 + a_idx)
        raw = predict_error_width(val, fitted)
        for weight in WEIGHTS:
            for scale in SCALES:
                rec = candidate_row(target, len(train), val, val_year, raw, alpha, weight, scale)
                if rec is not None:
                    rows.append(rec)
    return rows


def run_cv(rows: pd.DataFrame) -> pd.DataFrame:
    rows = DIW.attach_base_hw(rows[rows["target_id"].isin(TARGET_IDS)].copy())
    out: list[dict[str, Any]] = []
    for target in TARGETS:
        tid = str(target["target_id"])
        target_rows = rows[rows["target_id"].eq(tid)].reset_index(drop=True)
        for val_year in VAL_YEARS:
            train = target_rows[target_rows["origin_year"] < int(val_year)].reset_index(drop=True)
            val = target_rows[target_rows["origin_year"].eq(int(val_year))].reset_index(drop=True)
            if train.empty or val.empty:
                continue
            print(f"[cv] {target['display']} val_year={val_year} train={len(train):,} val={len(val):,}", flush=True)
            out.extend(evaluate_target_fold(target, train, val, val_year))
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
            pred_hw_p10=("pred_hw_p10", "mean"),
            pred_hw_p90=("pred_hw_p90", "mean"),
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        target = TARGET_BY_ID[str(row["target_id"])]
        gates_cfg = dict(target["gates"])
        gates_cfg.update(DIW.WIDTH_GATE_OVERRIDES.get(str(row["target_id"]), {}))
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


def parse_candidate(candidate: str) -> dict[str, float]:
    parts = dict(item.split("=", 1) for item in candidate.split("|")[1:])
    return {"alpha": float(parts["alpha"]), "weight": float(parts["w"]), "scale": float(parts["scale"])}


def load_inference_context(region: str, window: int) -> tuple[pd.DataFrame, list[str]]:
    cols = FR.allowed_feature_columns(region)
    feat = pd.read_parquet(FEATURES / f"inference_window_{window}_{region}.parquet", columns=["latitude", "longitude"] + cols)
    feat["latitude"] = pd.to_numeric(feat["latitude"], errors="coerce").astype("float64").round(2)
    feat["longitude"] = pd.to_numeric(feat["longitude"], errors="coerce").astype("float64").round(2)
    rename = {c: f"ctx_{c}" for c in cols}
    feat = feat.rename(columns=rename)
    return feat[["latitude", "longitude"] + list(rename.values())], list(rename.values())


def add_hres_fields(part: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    out = part.copy()
    horizon = int(target["horizon"])
    lead = hres_lead(horizon)
    hres_center = np.full(len(out), np.nan, dtype="float64")
    hres_speed = np.full(len(out), np.nan, dtype="float64")
    for (level, hour), idx in out.groupby(["level", "hour"], sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        hr = int(hour)
        if str(level) in ("10m", "100m"):
            d_col = f"ctx_fcst_dir_d{lead}_h{hr}"
            s_col = f"ctx_fcst_speed_d{lead}_h{hr}"
            hres_center[idx_arr] = pd.to_numeric(out.loc[idx_arr, d_col], errors="coerce").to_numpy(dtype="float64") % 360.0
            spd = pd.to_numeric(out.loc[idx_arr, s_col], errors="coerce").to_numpy(dtype="float64")
            if str(level) == "100m":
                spd = spd * 1.25
            hres_speed[idx_arr] = spd
        else:
            u_col = f"ctx_fcst_u_{level}_d{lead}_h{hr}"
            v_col = f"ctx_fcst_v_{level}_d{lead}_h{hr}"
            u = pd.to_numeric(out.loc[idx_arr, u_col], errors="coerce").to_numpy(dtype="float64")
            v = pd.to_numeric(out.loc[idx_arr, v_col], errors="coerce").to_numpy(dtype="float64")
            hres_center[idx_arr] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
            hres_speed[idx_arr] = np.sqrt(u * u + v * v)
    out["hres_center"] = hres_center
    out["hres_speed"] = hres_speed
    out["base_hres_delta"] = DIW.circ_abs_diff(out["base_center"].to_numpy(dtype="float64"), hres_center)
    out["hres_dir_sector"] = RNS.direction_sector(hres_center)
    return out


def inference_rows_for_target(df: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    idx = df.index[DIW.submission_target_mask(df, target)]
    parts: list[pd.DataFrame] = []
    for window, sub_idx in df.loc[idx].groupby("window", sort=False).groups.items():
        window = int(window)
        part = df.loc[np.asarray(list(sub_idx), dtype=int)].copy()
        part["_submission_index"] = part.index.to_numpy(dtype="int64")
        ctx, _ = load_inference_context(str(target["region"]), window)
        part["latitude"] = pd.to_numeric(part["latitude"], errors="coerce").astype("float64").round(2)
        part["longitude"] = pd.to_numeric(part["longitude"], errors="coerce").astype("float64").round(2)
        part = part.merge(ctx, on=["latitude", "longitude"], how="left", validate="many_to_one", sort=False)
        meta = json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text(encoding="utf-8"))
        origin = pd.Timestamp(meta["context_end"])
        part["target_id"] = str(target["target_id"])
        part["display"] = str(target["display"])
        part["problem"] = "dir"
        part["group"] = str(target["group"])
        part["origin_year"] = int(origin.year)
        part["origin_time"] = str(origin)
        part["month"] = int(origin.month)
        part["season"] = season_from_month(int(origin.month))
        part["base_center"] = pd.to_numeric(part["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        part["base_lo"] = pd.to_numeric(part["dir_05"], errors="coerce").to_numpy(dtype="float64") % 360.0
        part["base_hi"] = pd.to_numeric(part["dir_95"], errors="coerce").to_numpy(dtype="float64") % 360.0
        part["base_hw"] = ((part["base_hi"] - part["base_lo"]) % 360.0) / 2.0
        part = add_hres_fields(part, target)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out


def apply_model_to_submission(df: pd.DataFrame, target: dict[str, Any], train: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    params = parse_candidate(str(selected["candidate"]))
    fitted = fit_error_model(train, params["alpha"], seed=20260614 + 999)
    inf = inference_rows_for_target(df, target)
    raw = predict_error_width(inf, fitted)
    base_hw = inf["base_hw"].to_numpy(dtype="float64")
    pred_hw = np.clip((1.0 - params["weight"]) * base_hw + params["weight"] * raw * params["scale"], 5.0, 179.9)
    idx = inf["_submission_index"].to_numpy(dtype="int64")
    center = inf["base_center"].to_numpy(dtype="float64") % 360.0
    old_lo = inf["base_lo"].to_numpy(dtype="float64") % 360.0
    old_hi = inf["base_hi"].to_numpy(dtype="float64") % 360.0
    new_lo = (center - pred_hw) % 360.0
    new_hi = (center + pred_hw) % 360.0
    changed = (np.round(DIW.circ_abs_diff(new_lo, old_lo), 2) > 0.0) | (np.round(DIW.circ_abs_diff(new_hi, old_hi), 2) > 0.0)
    df.loc[idx, "dir_05"] = new_lo
    df.loc[idx, "dir_95"] = new_hi
    return {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "candidate": str(selected["candidate"]),
        "rows_in_scope": int(len(idx)),
        "rows_changed": int(changed.sum()),
        "new_half_width_median": float(np.nanmedian(pred_hw)),
        "new_half_width_p10": float(np.nanquantile(pred_hw, 0.10)),
        "new_half_width_p90": float(np.nanquantile(pred_hw, 0.90)),
        "width_move_p90": float(np.nanquantile(np.abs(pred_hw - base_hw), 0.90)),
    }


def write_submission(selected: list[dict[str, Any]], all_rows: pd.DataFrame) -> dict[str, Any]:
    df = pd.read_csv(BASE_CSV, low_memory=False)
    patches: list[dict[str, Any]] = []
    enriched = DIW.attach_base_hw(all_rows[all_rows["target_id"].isin(TARGET_IDS)].copy())
    for row in selected:
        target = TARGET_BY_ID[str(row["target_id"])]
        train = enriched[enriched["target_id"].eq(str(row["target_id"]))].reset_index(drop=True)
        patches.append(apply_model_to_submission(df, target, train, row))
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
            "alphas": list(ALPHAS),
            "weights": list(WEIGHTS),
            "scales": list(SCALES),
            "target_gates": {
                str(t["target_id"]): {**t["gates"], **DIW.WIDTH_GATE_OVERRIDES.get(str(t["target_id"]), {})}
                for t in TARGETS
            },
            "public_feedback_use": "leaderboard snapshot is used only for target eligibility and safety thresholds",
        },
        "competition_rule_notes": [
            "Uses only official phase1 training features/reanalysis labels, current generated base predictions, and official inference feature files.",
            "No external data or hidden/scoring-server labels are used.",
            "Public leaderboard values are aggregate gates only and are never row-level training labels or features.",
            "Any emitted submission changes only direction interval widths dir_05/dir_95 around existing dir_50 centers.",
        ],
        "code_hashes": {
            "builder": sha256(Path(__file__).resolve()),
            "runner": sha256(ROOT / "run_dir_error_width_newsignal_v1_e2e.ps1"),
            "feature_rich_builder": sha256(ROOT / "build_feature_rich_newsignal_v1_candidate.py"),
            "interval_builder": sha256(ROOT / "build_dir_interval_newsignal_v1_candidate.py"),
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
        "## Branch Result - Direction Error-Width New Signal V1",
        "",
        f"- Runner: `run_dir_error_width_newsignal_v1_e2e.ps1`.",
        f"- Builder: `build_dir_error_width_newsignal_v1_candidate.py`.",
        f"- Manifest: `{MANIFEST}`.",
        f"- Decisions: `{DECISION_CSV}`.",
        f"- Fold rows: `{CV_BY_FOLD_CSV}`.",
        f"- Submission zip: `{OUT_ZIP if OUT_ZIP.exists() else 'none written'}`.",
        f"- Status: `{status}`.",
        f"- Candidates evaluated: `{len(decisions)}`.",
        f"- CV-passing candidates: `{len(selected)}`.",
        "- Method: LightGBM quantile model for conditional circular absolute error; centers remain locked.",
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
        lines.append("- Decision: a gated row-level error-width candidate was emitted for scoring.")
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
        raise SystemExit(f"Missing enriched row cache: {ROW_CACHE}. Run run_feature_rich_newsignal_v1_e2e.ps1 first.")

    rows = pd.read_parquet(ROW_CACHE)
    rows = rows[rows["target_id"].isin(TARGET_IDS)].reset_index(drop=True)
    rows = DIW.attach_base_hw(rows)
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
                "reason": "No row-level direction error-width candidate cleared strict public-gap, mean-gain, worst-fold, regime, score-ceiling, width-movement, and coverage gates.",
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
            "reason": "At least one row-level direction error-width policy cleared strict CV gates and was applied with centers locked.",
            "candidates_evaluated": int(len(decisions)),
            "selected": selected,
            "submission": output,
        },
    )
    append_decision_log(status, decisions, selected)


if __name__ == "__main__":
    main()
