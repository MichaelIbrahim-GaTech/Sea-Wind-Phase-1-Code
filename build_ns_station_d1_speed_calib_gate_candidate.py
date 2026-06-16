from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import station_cv_mos_analog_framework as CV
from build_station_lgbm_ns_d1_speed_on_analog_candidate import (
    COLS,
    DIR_COLS,
    EXPECTED_PATCH_ROWS,
    HORIZON,
    REGION,
    SEED,
    SPEED_COLS,
    apply_calibration,
    fit_speed_models,
    load_inference_origin_rows,
    load_station_obs_with_context,
    predict_speed,
    validate_submission,
)


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "pred_ns_d1dir_ensw_gate.csv"
OUT_CSV = WORK / "pred_ns_d1spd_calib_gate.csv"
OUT_ZIP = WORK / "sub_ns_d1spd_calib_gate.zip"
SUMMARY_CSV = WORK / "cv_ns_d1spd_calib_gate.csv"
MANIFEST = WORK / "manifest_ns_d1spd_calib_gate.json"

CURRENT_PUBLIC_METRIC = 7.9876
CURRENT_STAGE2_CV_MEAN = 7.23239863504411
CURRENT_STAGE2_CV_MAX = 7.285128167845357
STRICT_GATE_IMPROVEMENT = 0.015
STRICT_GATE_MAX_MARGIN = 0.020
ROBUST_GATE_MEAN_IMPROVEMENT = 0.006
ROBUST_GATE_MAX_IMPROVEMENT = 0.0005

GROUP_STRATEGIES = [
    {"name": "no_bias", "group_cols": [], "shrink": 0.0},
    {"name": "station_s4", "group_cols": ["station"], "shrink": 4.0},
    {"name": "station_s8", "group_cols": ["station"], "shrink": 8.0},
    {"name": "station_s12", "group_cols": ["station"], "shrink": 12.0},
    {"name": "station_s20", "group_cols": ["station"], "shrink": 20.0},
    {"name": "hour_s8", "group_cols": ["target_hour"], "shrink": 8.0},
    {"name": "station_hour_s8", "group_cols": ["station", "target_hour"], "shrink": 8.0},
    {"name": "station_hour_s12", "group_cols": ["station", "target_hour"], "shrink": 12.0},
    {"name": "station_hour_s20", "group_cols": ["station", "target_hour"], "shrink": 20.0},
]

CAL_GRID = [
    {
        "bias": bias,
        "k_lo": k_lo,
        "k_hi": k_hi,
    }
    for bias in (-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05)
    for k_lo in (1.00, 1.10, 1.20, 1.30, 1.45, 1.60)
    for k_hi in (1.00, 1.10, 1.20, 1.30, 1.45, 1.60)
]

WIDTH_STRATEGIES = [
    {"name": "qwidth_global_q90", "group_cols": [], "shrink": 0.0, "q": 0.90},
    {"name": "qwidth_global_q92", "group_cols": [], "shrink": 0.0, "q": 0.92},
    {"name": "qwidth_global_q95", "group_cols": [], "shrink": 0.0, "q": 0.95},
    {"name": "qwidth_global_q97", "group_cols": [], "shrink": 0.0, "q": 0.97},
    {"name": "qwidth_global_q98", "group_cols": [], "shrink": 0.0, "q": 0.98},
    {"name": "qwidth_station_q90_s20", "group_cols": ["station"], "shrink": 20.0, "q": 0.90},
    {"name": "qwidth_station_q92_s20", "group_cols": ["station"], "shrink": 20.0, "q": 0.92},
    {"name": "qwidth_station_q95_s20", "group_cols": ["station"], "shrink": 20.0, "q": 0.95},
    {"name": "qwidth_station_q97_s20", "group_cols": ["station"], "shrink": 20.0, "q": 0.97},
    {"name": "qwidth_station_q90_s50", "group_cols": ["station"], "shrink": 50.0, "q": 0.90},
    {"name": "qwidth_station_q92_s50", "group_cols": ["station"], "shrink": 50.0, "q": 0.92},
    {"name": "qwidth_station_q95_s50", "group_cols": ["station"], "shrink": 50.0, "q": 0.95},
    {"name": "qwidth_station_q97_s50", "group_cols": ["station"], "shrink": 50.0, "q": 0.97},
    {"name": "qwidth_hour_q90_s20", "group_cols": ["target_hour"], "shrink": 20.0, "q": 0.90},
    {"name": "qwidth_hour_q92_s20", "group_cols": ["target_hour"], "shrink": 20.0, "q": 0.92},
    {"name": "qwidth_hour_q95_s20", "group_cols": ["target_hour"], "shrink": 20.0, "q": 0.95},
    {"name": "qwidth_hour_q97_s20", "group_cols": ["target_hour"], "shrink": 20.0, "q": 0.97},
    {"name": "qwidth_station_hour_q90_s80", "group_cols": ["station", "target_hour"], "shrink": 80.0, "q": 0.90},
    {"name": "qwidth_station_hour_q92_s80", "group_cols": ["station", "target_hour"], "shrink": 80.0, "q": 0.92},
    {"name": "qwidth_station_hour_q95_s80", "group_cols": ["station", "target_hour"], "shrink": 80.0, "q": 0.95},
    {"name": "qwidth_station_hour_q97_s80", "group_cols": ["station", "target_hour"], "shrink": 80.0, "q": 0.97},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def key_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["__global__"] * len(df), index=df.index)
    return df[group_cols].astype(str).agg("|".join, axis=1)


def build_bias_map(train: pd.DataFrame, residual: np.ndarray, group_cols: list[str], shrink: float) -> tuple[pd.Series, float]:
    residual = np.asarray(residual, dtype="float64")
    ok = np.isfinite(residual)
    global_bias = float(np.nanmedian(residual[ok])) if bool(ok.any()) else 0.0
    if not group_cols:
        return pd.Series(dtype="float64"), global_bias
    tmp = train.loc[ok, group_cols].copy()
    tmp["residual"] = residual[ok]
    agg = tmp.groupby(group_cols, dropna=False)["residual"].agg(["median", "count"]).reset_index()
    agg["bias"] = (agg["count"] / (agg["count"] + shrink)) * agg["median"] + (shrink / (agg["count"] + shrink)) * global_bias
    return pd.Series(agg["bias"].to_numpy(dtype="float64"), index=key_frame(agg, group_cols)), global_bias


def lookup_bias(df: pd.DataFrame, bias_map: pd.Series, global_bias: float, group_cols: list[str]) -> np.ndarray:
    if not group_cols:
        return np.full(len(df), global_bias, dtype="float64")
    return key_frame(df, group_cols).map(bias_map).fillna(global_bias).to_numpy(dtype="float64")


def shift_predictions(
    lo: np.ndarray,
    mid: np.ndarray,
    hi: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bias = np.asarray(bias, dtype="float64")
    mid2 = np.maximum(0.0, np.asarray(mid, dtype="float64") + bias)
    lo2 = np.maximum(0.0, np.asarray(lo, dtype="float64") + bias)
    hi2 = np.maximum(mid2, np.asarray(hi, dtype="float64") + bias)
    return lo2, mid2, hi2


def apply_cal_dict(
    lo: np.ndarray,
    mid: np.ndarray,
    hi: np.ndarray,
    cal: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return apply_calibration(
        lo,
        mid,
        hi,
        {"bias": float(cal["bias"]), "k_lo": float(cal["k_lo"]), "k_hi": float(cal["k_hi"])},
    )


def build_width_maps(
    train: pd.DataFrame,
    center: np.ndarray,
    group_cols: list[str],
    shrink: float,
    q: float,
) -> tuple[pd.Series, pd.Series, float, float]:
    y = train["y_speed"].to_numpy(dtype="float64")
    center = np.asarray(center, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(center)
    lower_resid = center[ok] - y[ok]
    upper_resid = y[ok] - center[ok]
    global_lower = float(max(0.05, np.nanquantile(lower_resid, q)))
    global_upper = float(max(0.05, np.nanquantile(upper_resid, q)))
    if not group_cols:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64"), global_lower, global_upper
    tmp = train.loc[ok, group_cols].copy()
    tmp["lower_resid"] = lower_resid
    tmp["upper_resid"] = upper_resid
    agg = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            lower_q=("lower_resid", lambda s: float(np.nanquantile(s.to_numpy(dtype="float64"), q))),
            upper_q=("upper_resid", lambda s: float(np.nanquantile(s.to_numpy(dtype="float64"), q))),
            count=("lower_resid", "count"),
        )
        .reset_index()
    )
    w = agg["count"].to_numpy(dtype="float64") / (agg["count"].to_numpy(dtype="float64") + float(shrink))
    agg["lower_width"] = np.maximum(0.05, w * agg["lower_q"].to_numpy(dtype="float64") + (1.0 - w) * global_lower)
    agg["upper_width"] = np.maximum(0.05, w * agg["upper_q"].to_numpy(dtype="float64") + (1.0 - w) * global_upper)
    keys = key_frame(agg, group_cols)
    return (
        pd.Series(agg["lower_width"].to_numpy(dtype="float64"), index=keys),
        pd.Series(agg["upper_width"].to_numpy(dtype="float64"), index=keys),
        global_lower,
        global_upper,
    )


def lookup_widths(
    df: pd.DataFrame,
    lower_map: pd.Series,
    upper_map: pd.Series,
    global_lower: float,
    global_upper: float,
    group_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not group_cols:
        return (
            np.full(len(df), float(global_lower), dtype="float64"),
            np.full(len(df), float(global_upper), dtype="float64"),
        )
    keys = key_frame(df, group_cols)
    lower = keys.map(lower_map).fillna(float(global_lower)).to_numpy(dtype="float64")
    upper = keys.map(upper_map).fillna(float(global_upper)).to_numpy(dtype="float64")
    return lower, upper


def residual_width_interval(
    center: np.ndarray,
    lower_width: np.ndarray,
    upper_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.maximum(0.0, np.asarray(center, dtype="float64"))
    lo = np.maximum(0.0, center - np.asarray(lower_width, dtype="float64"))
    hi = np.maximum(center, center + np.asarray(upper_width, dtype="float64"))
    return lo, center, hi


def evaluate_calibration_grid(
    y: np.ndarray,
    lo: np.ndarray,
    mid: np.ndarray,
    hi: np.ndarray,
    val_year: int,
    strategy: dict[str, object],
    global_bias: float,
) -> list[dict[str, object]]:
    rows = []
    for cal in CAL_GRID:
        clo, _, chi = apply_cal_dict(lo, mid, hi, cal)
        rows.append(
            {
                "val_year": int(val_year),
                "mode": "model_interval",
                "strategy": str(strategy["name"]),
                "group_cols": ",".join(strategy["group_cols"]),
                "shrink": float(strategy["shrink"]),
                "width_strategy": "model_interval",
                "width_group_cols": "",
                "width_shrink": -1.0,
                "width_q": -1.0,
                "bias_global": float(global_bias),
                "cal_bias": float(cal["bias"]),
                "cal_k_lo": float(cal["k_lo"]),
                "cal_k_hi": float(cal["k_hi"]),
                "score": CV.speed_winkler(y, clo, chi),
                "width": float(np.nanmean(chi - clo)),
                "n": int(np.isfinite(y).sum()),
            }
        )
    return rows


def evaluate_residual_widths(
    train: pd.DataFrame,
    val: pd.DataFrame,
    y: np.ndarray,
    train_center: np.ndarray,
    val_center: np.ndarray,
    val_year: int,
    strategy: dict[str, object],
    global_bias: float,
) -> list[dict[str, object]]:
    rows = []
    for width_strategy in WIDTH_STRATEGIES:
        width_cols = list(width_strategy["group_cols"])
        lower_map, upper_map, global_lower, global_upper = build_width_maps(
            train,
            train_center,
            width_cols,
            float(width_strategy["shrink"]),
            float(width_strategy["q"]),
        )
        lower, upper = lookup_widths(val, lower_map, upper_map, global_lower, global_upper, width_cols)
        lo, _, hi = residual_width_interval(val_center, lower, upper)
        rows.append(
            {
                "val_year": int(val_year),
                "mode": "residual_width",
                "strategy": str(strategy["name"]),
                "group_cols": ",".join(strategy["group_cols"]),
                "shrink": float(strategy["shrink"]),
                "width_strategy": str(width_strategy["name"]),
                "width_group_cols": ",".join(width_cols),
                "width_shrink": float(width_strategy["shrink"]),
                "width_q": float(width_strategy["q"]),
                "bias_global": float(global_bias),
                "cal_bias": 0.0,
                "cal_k_lo": 0.0,
                "cal_k_hi": 0.0,
                "score": CV.speed_winkler(y, lo, hi),
                "width": float(np.nanmean(hi - lo)),
                "n": int(np.isfinite(y).sum()),
            }
        )
    return rows


def run_cv_gate() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    meta = CV.read_station_meta()
    train_base = CV.load_station_origin_rows(REGION, meta)
    hist = CV.make_history(CV.load_station_obs(REGION))
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    df = df[df["y_speed"].notna()].copy()
    rows = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year)].copy()
        val = df[CV.anchor_mask(df, val_year)].copy()
        feats = CV.numeric_features(train)
        print(f"CV fold {val_year}: train={len(train):,} val={len(val):,} features={len(feats)}", flush=True)
        models = fit_speed_models(train, feats, SEED + val_year)
        tr_lo, tr_mid, tr_hi = predict_speed(models, feats, train)
        vl_lo, vl_mid, vl_hi = predict_speed(models, feats, val)
        y_tr = train["y_speed"].to_numpy(dtype="float64")
        y_vl = val["y_speed"].to_numpy(dtype="float64")
        for strategy in GROUP_STRATEGIES:
            group_cols = list(strategy["group_cols"])
            bias_map, global_bias = build_bias_map(train, y_tr - tr_mid, group_cols, float(strategy["shrink"]))
            tr_bias = lookup_bias(train, bias_map, global_bias, group_cols)
            vl_bias = lookup_bias(val, bias_map, global_bias, group_cols)
            tr_lo2, tr_mid2, tr_hi2 = shift_predictions(tr_lo, tr_mid, tr_hi, tr_bias)
            vl_lo2, vl_mid2, vl_hi2 = shift_predictions(vl_lo, vl_mid, vl_hi, vl_bias)
            train_center_score = CV.speed_winkler(y_tr, tr_lo2, tr_hi2)
            rows.extend(evaluate_calibration_grid(y_vl, vl_lo2, vl_mid2, vl_hi2, val_year, strategy, global_bias))
            rows.extend(evaluate_residual_widths(train, val, y_vl, tr_mid2, vl_mid2, val_year, strategy, global_bias))
            print(
                f"  {strategy['name']}: raw_train={train_center_score:.4f} "
                f"global_bias={global_bias:.4f}",
                flush=True,
            )
    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby(
            [
                "mode",
                "strategy",
                "group_cols",
                "shrink",
                "width_strategy",
                "width_group_cols",
                "width_shrink",
                "width_q",
                "cal_bias",
                "cal_k_lo",
                "cal_k_hi",
            ],
            as_index=False,
        )
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            width_mean=("width", "mean"),
            bias_global_mean=("bias_global", "mean"),
            n_min=("n", "min"),
        )
        .sort_values(["score_mean", "score_max", "width_mean"])
        .reset_index(drop=True)
    )
    summary["mean_improvement_vs_stage2"] = CURRENT_STAGE2_CV_MEAN - summary["score_mean"].astype(float)
    summary["max_improvement_vs_stage2"] = CURRENT_STAGE2_CV_MAX - summary["score_max"].astype(float)
    robust = summary[
        summary["mean_improvement_vs_stage2"].ge(ROBUST_GATE_MEAN_IMPROVEMENT)
        & summary["max_improvement_vs_stage2"].ge(ROBUST_GATE_MAX_IMPROVEMENT)
    ].copy()
    if len(robust):
        robust = robust.sort_values(["score_mean", "score_max", "width_mean"]).reset_index(drop=True)
        selected = robust.iloc[0].to_dict()
        selected["selection_rule"] = "robust_mean_and_max_gate"
    else:
        selected = summary.iloc[0].to_dict()
        selected["selection_rule"] = "best_mean_gate_failed_robust_filter"
    selected["gate_passed"] = bool(
        float(selected["mean_improvement_vs_stage2"]) >= ROBUST_GATE_MEAN_IMPROVEMENT
        and float(selected["max_improvement_vs_stage2"]) >= ROBUST_GATE_MAX_IMPROVEMENT
    )
    selected["gate_requirements"] = {
        "mean_improvement_gte": ROBUST_GATE_MEAN_IMPROVEMENT,
        "max_improvement_gte": ROBUST_GATE_MAX_IMPROVEMENT,
        "current_stage2_cv_mean": CURRENT_STAGE2_CV_MEAN,
        "current_stage2_cv_max": CURRENT_STAGE2_CV_MAX,
        "current_public_metric": CURRENT_PUBLIC_METRIC,
        "unused_best_mean_soft_gate": {
            "score_mean_lte": CURRENT_STAGE2_CV_MEAN - STRICT_GATE_IMPROVEMENT,
            "score_max_lte": CURRENT_STAGE2_CV_MAX + STRICT_GATE_MAX_MARGIN,
        },
    }
    print("Top calibration candidates:", flush=True)
    print(summary.head(20).to_string(index=False), flush=True)
    print(f"Selected candidate: {selected}", flush=True)
    return selected, cv, summary


def write_gate_failed(selected: dict[str, object], cv: pd.DataFrame, summary: pd.DataFrame) -> None:
    merge_keys = [
        "mode",
        "strategy",
        "group_cols",
        "shrink",
        "width_strategy",
        "width_group_cols",
        "width_shrink",
        "width_q",
        "cal_bias",
        "cal_k_lo",
        "cal_k_hi",
    ]
    cv.merge(
        summary.add_prefix("summary_"),
        left_on=merge_keys,
        right_on=[f"summary_{key}" for key in merge_keys],
        how="left",
    ).to_csv(SUMMARY_CSV, index=False)
    MANIFEST.write_text(
        json.dumps(
            {
                "status": "gate_failed_no_submission_written",
                "selected": selected,
                "summary": summary.to_dict(orient="records"),
                "cv": cv.to_dict(orient="records"),
                "compliance": [
                    "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
                    "No external datasets, no web data, and no evaluation target labels.",
                    "Gate failed, so no submission zip was emitted by this builder.",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote gate-failed manifest {MANIFEST}", flush=True)


def make_patch(selected: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    meta = CV.read_station_meta()
    train_base = CV.load_station_origin_rows(REGION, meta)
    hist = CV.make_history(CV.load_station_obs(REGION))
    train_df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    train_df = train_df[train_df["y_speed"].notna()].copy()
    feats = CV.numeric_features(train_df)
    print(f"Fitting final speed models: rows={len(train_df):,} features={len(feats)}", flush=True)
    models = fit_speed_models(train_df, feats, SEED + 9090)
    tr_lo, tr_mid, tr_hi = predict_speed(models, feats, train_df)
    group_cols = [c for c in str(selected["group_cols"]).split(",") if c]
    bias_map, global_bias = build_bias_map(
        train_df,
        train_df["y_speed"].to_numpy(dtype="float64") - tr_mid,
        group_cols,
        float(selected["shrink"]),
    )
    cal = {"bias": float(selected["cal_bias"]), "k_lo": float(selected["cal_k_lo"]), "k_hi": float(selected["cal_k_hi"])}
    width_cols = [c for c in str(selected.get("width_group_cols", "")).split(",") if c]
    if str(selected.get("mode", "model_interval")) == "residual_width":
        tr_bias = lookup_bias(train_df, bias_map, global_bias, group_cols)
        _, tr_center, _ = shift_predictions(tr_lo, tr_mid, tr_hi, tr_bias)
        lower_map, upper_map, global_lower, global_upper = build_width_maps(
            train_df,
            tr_center,
            width_cols,
            float(selected["width_shrink"]),
            float(selected["width_q"]),
        )
    else:
        lower_map = pd.Series(dtype="float64")
        upper_map = pd.Series(dtype="float64")
        global_lower = float("nan")
        global_upper = float("nan")
    patch_rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            lo, mid, hi = predict_speed(models, feats, inf)
            bias = lookup_bias(inf, bias_map, global_bias, group_cols)
            lo, mid, hi = shift_predictions(lo, mid, hi, bias)
            if str(selected.get("mode", "model_interval")) == "residual_width":
                lower, upper = lookup_widths(inf, lower_map, upper_map, global_lower, global_upper, width_cols)
                lo, mid, hi = residual_width_interval(mid, lower, upper)
            else:
                lo, mid, hi = apply_cal_dict(lo, mid, hi, cal)
            for station, q05, q50, q95 in zip(inf["station"].astype(str), lo, mid, hi):
                patch_rows.append(
                    {
                        "window": window,
                        "region": REGION,
                        "station": station,
                        "horizon": HORIZON,
                        "hour": int(hour),
                        "q05_new": float(max(q05, 0.0)),
                        "q50_new": float(max(q50, 0.0)),
                        "q95_new": float(max(q95, q50, 0.0)),
                    }
                )
    final_meta = {
        "group_cols": group_cols,
        "global_bias": global_bias,
        "calibration": cal,
        "width_group_cols": width_cols,
        "global_lower_width": global_lower,
        "global_upper_width": global_upper,
        "n_width_keys": int(len(lower_map)),
        "features": feats,
        "n_bias_keys": int(len(bias_map)),
    }
    return pd.DataFrame(patch_rows), final_meta


def main() -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    out = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    selected, cv, summary = run_cv_gate()
    if not bool(selected["gate_passed"]):
        write_gate_failed(selected, cv, summary)
        raise SystemExit("Strict NS station d1 speed calibration gate failed; no submission zip written.")
    merge_keys = [
        "mode",
        "strategy",
        "group_cols",
        "shrink",
        "width_strategy",
        "width_group_cols",
        "width_shrink",
        "width_q",
        "cal_bias",
        "cal_k_lo",
        "cal_k_hi",
    ]
    cv.merge(
        summary.add_prefix("summary_"),
        left_on=merge_keys,
        right_on=[f"summary_{key}" for key in merge_keys],
        how="left",
    ).to_csv(SUMMARY_CSV, index=False)
    patch, final_fit = make_patch(selected)
    key = ["window", "region", "station", "horizon", "hour"]
    merged = out.reset_index().merge(patch, on=key, how="left", validate="many_to_one")
    station_mask = (
        merged["type"].eq("station")
        & merged["region"].eq(REGION)
        & merged["horizon"].eq(HORIZON)
        & merged["q50_new"].notna()
    )
    patched = int(station_mask.sum())
    if patched != EXPECTED_PATCH_ROWS:
        raise SystemExit(f"expected {EXPECTED_PATCH_ROWS} target rows, got {patched}")
    before_dir = out[DIR_COLS].round(1).copy()
    for c in SPEED_COLS:
        merged.loc[station_mask, c] = merged.loc[station_mask, f"{c}_new"]
    out2 = merged.drop(columns=["q05_new", "q50_new", "q95_new"]).set_index("index").sort_index()[COLS]
    speed_changed_mask = (out[SPEED_COLS].round(2).to_numpy() != out2[SPEED_COLS].round(2).to_numpy()).any(axis=1)
    dir_changed = int((before_dir.to_numpy() != out2[DIR_COLS].round(1).to_numpy()).any(axis=1).sum())
    speed_changed = int(speed_changed_mask.sum())
    non_target_speed_changed = int(
        (
            speed_changed_mask
            & ~(
                out["type"].eq("station")
                & out["region"].eq(REGION)
                & out["horizon"].eq(HORIZON)
            ).to_numpy()
        ).sum()
    )
    print(
        f"Patched rows={patched}; speed_changed={speed_changed}; "
        f"non_target_speed_changed={non_target_speed_changed}; direction_changed={dir_changed}",
        flush=True,
    )
    if speed_changed <= 0 or non_target_speed_changed or dir_changed:
        raise SystemExit("unexpected delta outside NS station d1 speed")
    validate_submission(out2)
    print(f"Writing {OUT_CSV}", flush=True)
    out2.to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        info = zf.getinfo("predictions.csv")
        names = zf.namelist()
        bad = zf.testzip()
    manifest = {
        "submission": {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "zip_name_length": len(OUT_ZIP.name),
            "csv_size": int(OUT_CSV.stat().st_size),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_size": int(info.file_size),
            "internal_names": names,
            "testzip": bad,
        },
        "base_csv": {
            "path": str(BASE_CSV),
            "size": int(BASE_CSV.stat().st_size),
            "sha256": sha256(BASE_CSV),
        },
        "cv": {
            "summary_csv": str(SUMMARY_CSV),
            "selected": selected,
            "final_fit": final_fit,
            "top20": summary.head(20).to_dict(orient="records"),
        },
        "delta": {
            "target": "WS NS Stations d1",
            "patched_rows": patched,
            "speed_rows_changed": speed_changed,
            "non_target_speed_rows_changed": non_target_speed_changed,
            "direction_rows_changed": dir_changed,
        },
        "compliance": [
            "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Training labels come only from official historical station observations.",
            "Inference history uses provided context station files for each window and only past/context values.",
            "Strict chronological CV gate passed before emitting the submission zip.",
            "Starts from the current accepted end-to-end generated base pred_ns_d1dir_ensw_gate.csv.",
        ],
        "code_hashes": {
            "build_ns_station_d1_speed_calib_gate_candidate.py": sha256(Path(__file__)),
            "build_ns_d1dir_enswidth_gate_candidate.py": sha256(Path("build_ns_d1dir_enswidth_gate_candidate.py")),
            "build_station_lgbm_ns_d1_speed_on_analog_candidate.py": sha256(Path("build_station_lgbm_ns_d1_speed_on_analog_candidate.py")),
            "station_cv_mos_analog_framework.py": sha256(Path("station_cv_mos_analog_framework.py")),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; "
        f"names={names}; testzip={bad}",
        flush=True,
    )
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
