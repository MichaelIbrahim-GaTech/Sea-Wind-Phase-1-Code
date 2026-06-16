from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_dir_error_width_newsignal_v1_candidate as DEW
import build_dir_interval_newsignal_v1_candidate as DIW
import build_feature_rich_newsignal_v1_candidate as FR
import build_regime_newsignal_v1_candidate as RNS
import direction_anchor_backtest as DAB
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

BASE_CSV = WORK / "pred_dir_error_width_newsignal_v1.csv"
ROW_CACHE = WORK / "gridlong_dir_error_width_v1_rows_s180.parquet"
OUT_CSV = WORK / "pred_dir_error_width_gridlong_v1.csv"
OUT_ZIP = WORK / "sub_direrrw_gridlong_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_dir_error_width_gridlong_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_dir_error_width_gridlong_v1_summary.csv"
DECISION_CSV = WORK / "decision_dir_error_width_gridlong_v1.csv"
MANIFEST = WORK / "manifest_dir_error_width_gridlong_v1.json"

ANCHOR_YEARS = (2019, 2020, 2021)
VAL_YEARS = (2020, 2021)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
SAMPLE_PER_ANCHOR_DATE = 180
ALPHAS = (0.80, 0.90, 0.95)
WEIGHTS = (0.25, 0.50, 0.75, 1.00)
SCALES = (0.70, 0.85, 1.00, 1.15, 1.30)

SURFACE_LEVELS = ("10m", "100m")
PRESSURE_LEVELS = DAB.PRESSURE_LEVELS
HOURS = DAB.HOURS


PUBLIC_GAPS: dict[str, dict[str, Any]] = {
    "dir_ecs_surface_d1": {"ours": 131.0613, "top_best": 112.82, "gap_to_best": 18.2413, "top_best_name": "JLShen"},
    "dir_ns_surface_d7": {"ours": 298.5943, "top_best": 256.44, "gap_to_best": 42.1543, "top_best_name": "sajayrrr"},
    "dir_ns_surface_d14": {"ours": 325.4191, "top_best": 298.76, "gap_to_best": 26.6591, "top_best_name": "JLShen"},
    "dir_ns_pressure_d7": {"ours": 280.1930, "top_best": 236.54, "gap_to_best": 43.6530, "top_best_name": "sajayrrr"},
    "dir_ns_pressure_d14": {"ours": 326.7063, "top_best": 300.28, "gap_to_best": 26.4263, "top_best_name": "Matteo"},
    "dir_ecs_surface_d7": {"ours": 278.4071, "top_best": 266.45, "gap_to_best": 11.9571, "top_best_name": "JLShen"},
    "dir_ecs_surface_d14": {"ours": 327.5396, "top_best": 303.76, "gap_to_best": 23.7796, "top_best_name": "Matteo"},
    "dir_ecs_pressure_d7": {"ours": 252.7414, "top_best": 214.45, "gap_to_best": 38.2914, "top_best_name": "sajayrrr"},
    "dir_ecs_pressure_d14": {"ours": 315.4798, "top_best": 285.90, "gap_to_best": 29.5798, "top_best_name": "sajayrrr"},
}


def target_dict(region: str, group: str, horizon: int) -> dict[str, Any]:
    reg = "ns" if region == "north_sea" else "ecs"
    levels = SURFACE_LEVELS if group == "surface" else PRESSURE_LEVELS
    target_id = f"dir_{reg}_{group}_d{int(horizon)}"
    region_name = "NS" if region == "north_sea" else "ECS"
    group_name = "Surface" if group == "surface" else "Pressure"
    public = PUBLIC_GAPS[target_id]
    long_horizon = int(horizon) >= 7
    small_gap = float(public["gap_to_best"]) < 15.0
    gates = {
        "min_public_gap": 8.0,
        "min_mean_gain": 2.0 if (small_gap or not long_horizon) else 4.0,
        "min_worst_gain": 0.50 if not long_horizon else 1.00,
        "min_regime_gain": 0.50 if not long_horizon else 1.00,
        "max_score_max": float(public["ours"]) + (22.0 if not long_horizon else 45.0),
        "max_move_p90": 70.0 if not long_horizon else 95.0,
        "max_move_p99": 110.0 if not long_horizon else 140.0,
        "max_changed_fraction": 1.00,
        "min_changed_fraction": 0.015,
        "min_val_selected": 100 if group == "surface" else 120,
        "min_train_selected": 100 if group == "surface" else 120,
        "max_infer_move_p90": 85.0 if not long_horizon else 105.0,
        "max_infer_move_mean": 45.0 if not long_horizon else 60.0,
    }
    return {
        "target_id": target_id,
        "display": f"Dir {region_name} {group_name} d{int(horizon)}",
        "problem": "dir",
        "region": region,
        "group": group,
        "levels": levels,
        "horizon": int(horizon),
        "gates": gates,
        "public": public,
    }


TARGETS: tuple[dict[str, Any], ...] = (
    target_dict("east_china_sea", "surface", 1),
    target_dict("north_sea", "surface", 7),
    target_dict("north_sea", "surface", 14),
    target_dict("north_sea", "pressure", 7),
    target_dict("north_sea", "pressure", 14),
    target_dict("east_china_sea", "surface", 7),
    target_dict("east_china_sea", "surface", 14),
    target_dict("east_china_sea", "pressure", 7),
    target_dict("east_china_sea", "pressure", 14),
)
TARGET_IDS = {str(t["target_id"]) for t in TARGETS}
TARGET_BY_ID = {str(t["target_id"]): t for t in TARGETS}


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


def anchor_dates(year: int) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{int(year)}-{mmdd}" for mmdd in ANCHOR_MMDD])


def hres_lead(horizon: int) -> int:
    return int(horizon) if int(horizon) in (1, 7) else 10


def hres_speed_from_features(df: pd.DataFrame, level: str, horizon: int, hour: int) -> np.ndarray:
    lead = hres_lead(int(horizon))
    if level in ("10m", "100m"):
        col = f"fcst_speed_d{lead}_h{int(hour)}"
        speed = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        if level == "100m":
            speed = speed * 1.25
        return speed
    u_col = f"fcst_u_{level}_d{lead}_h{int(hour)}"
    v_col = f"fcst_v_{level}_d{lead}_h{int(hour)}"
    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype="float64")
    return np.sqrt(u * u + v * v)


def sample_eval_rows(train: pd.DataFrame, year: int) -> pd.DataFrame:
    ev = train[train["time"].isin(set(anchor_dates(year)))].copy()
    parts: list[pd.DataFrame] = []
    for _, part in ev.groupby("time", sort=True):
        if SAMPLE_PER_ANCHOR_DATE > 0 and len(part) > SAMPLE_PER_ANCHOR_DATE:
            parts.append(part.sample(SAMPLE_PER_ANCHOR_DATE, random_state=20260614 + int(year)))
        else:
            parts.append(part)
    if not parts:
        raise SystemExit(f"No anchor rows for {year}")
    return pd.concat(parts, ignore_index=True)


def load_region_train(region: str, bundle: dict[str, Any], ctx_cols: list[str]) -> pd.DataFrame:
    cols = set(DAB.needed_feature_columns(bundle)).union(ctx_cols)
    for h in DAB.HORIZONS:
        for hour in HOURS:
            cols.add(f"dir_d{h}_h{hour}")
    for lead in (1, 7, 10):
        for hour in HOURS:
            cols.add(f"fcst_speed_d{lead}_h{hour}")
    train = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=sorted(cols))
    train["time"] = pd.to_datetime(train["time"])
    train["latitude"] = pd.to_numeric(train["latitude"], errors="coerce").astype("float64").round(2)
    train["longitude"] = pd.to_numeric(train["longitude"], errors="coerce").astype("float64").round(2)
    return train.reset_index(drop=True)


def build_region_rows(region: str, targets: list[dict[str, Any]]) -> pd.DataFrame:
    print(f"[rows] loading region={region}", flush=True)
    bundle = DAB.load_dir_bundle(region)
    ctx_cols = FR.allowed_feature_columns(region)
    train = load_region_train(region, bundle, ctx_cols)
    surf100 = DAB.load_surface100_lookup(region)
    pressure = DAB.load_pressure_lookup(region)
    out_parts: list[pd.DataFrame] = []
    for year in ANCHOR_YEARS:
        ev = sample_eval_rows(train, year)
        print(f"[rows] region={region} year={year} anchors={len(ev):,}", flush=True)
        centers = DAB.predict_model_centers(ev, bundle)
        ctx_pref = ev[ctx_cols].reset_index(drop=True).rename(columns={c: f"ctx_{c}" for c in ctx_cols})
        for target in targets:
            horizon = int(target["horizon"])
            for level in tuple(target["levels"]):
                native_hw = float(bundle["calibration"][str(level)][horizon]["half_width"])
                for hour in HOURS:
                    actual = DAB.target_direction(ev, str(level), horizon, int(hour), surf100, pressure)
                    base_center = centers[str(level)][(horizon, int(hour))]
                    hres = DAB.forecast_dir_from_features(ev, str(level), horizon, int(hour))
                    hres_center = np.asarray(hres, dtype="float64") if hres is not None else base_center
                    hspd = hres_speed_from_features(ev, str(level), horizon, int(hour))
                    base = pd.DataFrame(
                        {
                            "target_id": str(target["target_id"]),
                            "display": str(target["display"]),
                            "problem": "dir",
                            "region": region,
                            "group": str(target["group"]),
                            "level": str(level),
                            "horizon": horizon,
                            "hour": int(hour),
                            "origin_year": int(year),
                            "origin_time": ev["time"].astype(str).to_numpy(),
                            "month": ev["time"].dt.month.to_numpy(dtype="int16"),
                            "season": [RNS.season_from_month(int(m)) for m in ev["time"].dt.month.to_numpy(dtype="int16")],
                            "latitude": ev["latitude"].to_numpy(dtype="float64"),
                            "longitude": ev["longitude"].to_numpy(dtype="float64"),
                            "actual": actual,
                            "base_center": np.asarray(base_center, dtype="float64") % 360.0,
                            "base_hw": native_hw,
                            "hres_center": np.asarray(hres_center, dtype="float64") % 360.0,
                            "hres_speed": hspd,
                        }
                    )
                    base["base_hres_delta"] = DIW.circ_abs_diff(
                        base["base_center"].to_numpy(dtype="float64"),
                        base["hres_center"].to_numpy(dtype="float64"),
                    )
                    base["hres_dir_sector"] = RNS.direction_sector(base["hres_center"].to_numpy(dtype="float64"))
                    out_parts.append(pd.concat([base, ctx_pref], axis=1))
    return pd.concat(out_parts, ignore_index=True)


def ensure_row_cache() -> pd.DataFrame:
    if ROW_CACHE.exists():
        print(f"[cache] using {ROW_CACHE}", flush=True)
        return pd.read_parquet(ROW_CACHE)
    RNS.install_fast_anchor_predictors()
    parts: list[pd.DataFrame] = []
    for region in ("north_sea", "east_china_sea"):
        targets = [t for t in TARGETS if str(t["region"]) == region]
        parts.append(build_region_rows(region, targets))
    rows = pd.concat(parts, ignore_index=True)
    tmp = ROW_CACHE.with_suffix(ROW_CACHE.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    rows.to_parquet(tmp, index=False)
    if ROW_CACHE.exists():
        ROW_CACHE.unlink()
    tmp.replace(ROW_CACHE)
    print(f"[cache] wrote {ROW_CACHE} rows={len(rows):,}", flush=True)
    return rows


def evaluate_target_fold(target: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, val_year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [DEW.baseline_row(target, val, val_year)]
    for a_idx, alpha in enumerate(ALPHAS):
        print(f"  [fit] {target['display']} y={val_year} alpha={alpha:.2f}", flush=True)
        fitted = DEW.fit_error_model(train, alpha, seed=20260615 + int(val_year) * 100 + a_idx)
        raw = DEW.predict_error_width(val, fitted)
        for weight in WEIGHTS:
            for scale in SCALES:
                rec = DEW.candidate_row(target, len(train), val, val_year, raw, alpha, weight, scale)
                if rec is not None:
                    rows.append(rec)
    return rows


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
            print(f"[cv] {target['display']} val_year={val_year} train={len(train):,} val={len(val):,}", flush=True)
            out.extend(evaluate_target_fold(target, train, val, val_year))
    return pd.DataFrame(out)


def summarize_and_gate(folds: pd.DataFrame) -> pd.DataFrame:
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
        public = dict(target["public"])
        gates = {
            "not_baseline": str(row["candidate"]) != "current_model_native_width",
            "public_gap": float(public["gap_to_best"]) >= float(gates_cfg["min_public_gap"]),
            "mean_gain": float(row["gain"]) >= float(gates_cfg["min_mean_gain"]),
            "worst_gain": float(row["gain_min"]) >= float(gates_cfg["min_worst_gain"]),
            "regime_worst_gain": np.isfinite(float(row["regime_gain_min"])) and float(row["regime_gain_min"]) >= float(gates_cfg["min_regime_gain"]),
            "score_ceiling": float(row["score_max"]) <= float(gates_cfg["max_score_max"]),
            "cv_width_move_p90": float(row["move_p90"]) <= float(gates_cfg["max_move_p90"]),
            "cv_width_move_p99": float(row["move_p99"]) <= float(gates_cfg["max_move_p99"]),
            "changed_fraction": float(gates_cfg["min_changed_fraction"]) <= float(row["changed_fraction"]) <= float(gates_cfg["max_changed_fraction"]),
            "train_selected": int(row["train_selected_min"]) >= int(gates_cfg["min_train_selected"]),
            "val_selected": int(row["val_selected_min"]) >= int(gates_cfg["min_val_selected"]),
            "fold_count": int(row["fold_count"]) == len(VAL_YEARS),
        }
        out = dict(row)
        out.update(
            {
                "public_current": float(public["ours"]),
                "leader_reference": float(public["top_best"]),
                "public_gap": float(public["gap_to_best"]),
                "top_best_name": str(public.get("top_best_name", "")),
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


def apply_model_to_submission(df: pd.DataFrame, target: dict[str, Any], train: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    params = DEW.parse_candidate(str(selected["candidate"]))
    fitted = DEW.fit_error_model(train, params["alpha"], seed=20260615 + 999)
    inf = DEW.inference_rows_for_target(df, target)
    raw = DEW.predict_error_width(inf, fitted)
    base_hw = inf["base_hw"].to_numpy(dtype="float64")
    pred_hw = np.clip((1.0 - params["weight"]) * base_hw + params["weight"] * raw * params["scale"], 5.0, 179.9)
    idx = inf["_submission_index"].to_numpy(dtype="int64")
    center = inf["base_center"].to_numpy(dtype="float64") % 360.0
    old_lo = inf["base_lo"].to_numpy(dtype="float64") % 360.0
    old_hi = inf["base_hi"].to_numpy(dtype="float64") % 360.0
    new_lo = (center - pred_hw) % 360.0
    new_hi = (center + pred_hw) % 360.0
    changed = (np.round(DIW.circ_abs_diff(new_lo, old_lo), 2) > 0.0) | (np.round(DIW.circ_abs_diff(new_hi, old_hi), 2) > 0.0)
    move = np.abs(pred_hw - base_hw)
    audit = {
        "target_id": str(target["target_id"]),
        "display": str(target["display"]),
        "candidate": str(selected["candidate"]),
        "rows_in_scope": int(len(idx)),
        "rows_changed": int(changed.sum()),
        "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
        "new_half_width_median": float(np.nanmedian(pred_hw)),
        "new_half_width_p10": float(np.nanquantile(pred_hw, 0.10)),
        "new_half_width_p90": float(np.nanquantile(pred_hw, 0.90)),
        "width_move_mean": float(np.nanmean(move)),
        "width_move_p90": float(np.nanquantile(move, 0.90)),
    }
    gates = target["gates"]
    reasons = []
    if audit["width_move_mean"] > float(gates["max_infer_move_mean"]):
        reasons.append("infer_move_mean")
    if audit["width_move_p90"] > float(gates["max_infer_move_p90"]):
        reasons.append("infer_move_p90")
    if not (float(gates["min_changed_fraction"]) <= audit["changed_fraction"] <= float(gates["max_changed_fraction"])):
        reasons.append("infer_changed_fraction")
    audit["inference_gate_passed"] = not reasons
    audit["inference_reject_reasons"] = ",".join(reasons)
    if reasons:
        return audit
    df.loc[idx, "dir_05"] = new_lo
    df.loc[idx, "dir_95"] = new_hi
    return audit


def write_submission(selected: list[dict[str, Any]], all_rows: pd.DataFrame) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    df = pd.read_csv(BASE_CSV, low_memory=False)
    patches: list[dict[str, Any]] = []
    for row in selected:
        target = TARGET_BY_ID[str(row["target_id"])]
        train = all_rows[all_rows["target_id"].eq(str(row["target_id"]))].reset_index(drop=True)
        audit = apply_model_to_submission(df, target, train, row)
        patches.append(audit)
        if not audit.get("inference_gate_passed", False):
            return None, patches
    final = E2E.validate_final(df)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise RuntimeError(f"zip validation failed names={names} bad={bad}")
    return (
        {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "predictions_csv_size": int(info.file_size),
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_names": names,
            "testzip": bad,
            "patches": patches,
        },
        patches,
    )


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
            "anchor_years": list(ANCHOR_YEARS),
            "val_years": list(VAL_YEARS),
            "sample_per_anchor_date": SAMPLE_PER_ANCHOR_DATE,
            "alphas": list(ALPHAS),
            "weights": list(WEIGHTS),
            "scales": list(SCALES),
        },
        "competition_rule_notes": [
            "Uses official phase1 training features, official reanalysis targets, generated base predictions, and official inference feature files only.",
            "No external data or hidden/scoring-server labels are used.",
            "Public leaderboard values are aggregate target-priority and safety gates only; they are not row-level features or labels.",
            "Any emitted submission changes only direction interval widths dir_05/dir_95 around existing current-best dir_50 centers.",
        ],
        "code_hashes": {
            "builder": sha256(Path(__file__).resolve()),
            "runner": sha256(ROOT / "run_dir_error_width_gridlong_v1_e2e.ps1"),
            "v1_width_builder": sha256(ROOT / "build_dir_error_width_newsignal_v1_candidate.py"),
            "feature_rich_builder": sha256(ROOT / "build_feature_rich_newsignal_v1_candidate.py"),
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
        "## Branch Result - Direction Error-Width Grid Long V1",
        "",
        f"- Runner: `run_dir_error_width_gridlong_v1_e2e.ps1`.",
        f"- Builder: `build_dir_error_width_gridlong_v1_candidate.py`.",
        f"- Manifest: `{MANIFEST}`.",
        f"- Decisions: `{DECISION_CSV}`.",
        f"- Fold rows: `{CV_BY_FOLD_CSV}`.",
        f"- Submission zip: `{OUT_ZIP if OUT_ZIP.exists() else 'none written'}`.",
        f"- Status: `{status}`.",
        f"- Candidates evaluated: `{len(decisions)}`.",
        f"- CV-passing candidates: `{len(selected)}`.",
        "- Method: row-level LightGBM quantile model for conditional circular absolute error; centers remain locked.",
        "- Best candidates by target:",
    ]
    for row in top.to_dict("records"):
        lines.append(
            "  - `{display}`: `{candidate}`; mean gain `{gain:.4f}`, worst-fold gain `{gain_min:.4f}`, "
            "score max `{score_max:.4f}`; failed `{reject}`.".format(
                display=row["display"],
                candidate=row["candidate"],
                gain=float(row["gain"]),
                gain_min=float(row["gain_min"]),
                score_max=float(row["score_max"]),
                reject=str(row["reject_reasons"]) or "none",
            )
        )
    if selected:
        lines.append("- Decision: a gated grid-long direction width candidate was emitted for scoring.")
    else:
        lines.append("- Decision: no submission was emitted; current best remains `runs/v6_pressure_speed/sub_direrrw_v1.zip`.")
    with (ROOT / "submission_decisions.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cleanup_outputs()
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing promoted base CSV: {BASE_CSV}. Run/keep sub_direrrw_v1 first.")
    rows = ensure_row_cache()
    rows = rows[rows["target_id"].isin(TARGET_IDS)].reset_index(drop=True)
    folds = run_cv(rows)
    folds.to_csv(CV_BY_FOLD_CSV, index=False)
    decisions = summarize_and_gate(folds)
    decisions.to_csv(CV_SUMMARY_CSV, index=False)
    decisions.to_csv(DECISION_CSV, index=False)
    selected = select_candidates(decisions)
    if not selected:
        status = "blocked_no_submission"
        write_manifest(
            status,
            {
                "reason": "No remaining grid-direction row-level error-width candidate cleared strict public-gap, mean-gain, worst-fold, regime, score-ceiling, movement, and coverage gates.",
                "candidates_evaluated": int(len(decisions)),
                "selected": [],
                "top_by_target": decisions.groupby("display", sort=False).head(5).to_dict("records"),
            },
        )
        append_decision_log(status, decisions, selected)
        return

    output, patches = write_submission(selected, rows)
    if output is None:
        status = "blocked_inference_gate"
        write_manifest(
            status,
            {
                "reason": "At least one candidate cleared CV gates but failed inference movement/scope gates before any zip was written.",
                "candidates_evaluated": int(len(decisions)),
                "selected": selected,
                "patches": patches,
            },
        )
        append_decision_log(status, decisions, selected)
        return

    status = "submission_written"
    write_manifest(
        status,
        {
            "reason": "At least one remaining grid-direction row-level error-width policy cleared strict CV and inference gates and was applied with centers locked.",
            "candidates_evaluated": int(len(decisions)),
            "selected": selected,
            "submission": output,
        },
    )
    append_decision_log(status, decisions, selected)


if __name__ == "__main__":
    main()
