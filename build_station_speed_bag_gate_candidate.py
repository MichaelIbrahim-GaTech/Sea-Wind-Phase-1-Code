from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E
import station_cv_mos_analog_framework as CV
from build_ns_station_d1_speed_calib_gate_candidate import (
    apply_cal_dict,
    build_bias_map,
    build_width_maps,
    lookup_bias,
    lookup_widths,
    residual_width_interval,
    shift_predictions,
)
from build_station_lgbm_ns_d1_speed_on_analog_candidate import (
    load_inference_origin_rows,
    load_station_obs_with_context,
)


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_stndir_d1_bag_gate.csv"
OUT_CSV = WORK / "pred_stnspd_bag_gate.csv"
OUT_ZIP = WORK / "sub_stnspd_bag_gate.zip"
SUMMARY_CSV = WORK / "cv_stnspd_bag_gate.csv"
MANIFEST = WORK / "manifest_stnspd_bag_gate.json"

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS
HOURS = CV.HOURS
N_ESTIMATORS = 320

TARGETS = [
    {
        "id": "ns_d1",
        "label": "WS NS Stations d1",
        "region": "north_sea",
        "horizon": 1,
        "expected_rows": 256,
        "baseline_cv_mean": 7.225357343953551,
        "baseline_cv_max": 7.284335807394285,
        "required_mean_improvement": 0.0025,
        "max_margin": 0.0200,
        "public_current": 7.9448,
        "families": ["direct_bag2", "hres_resid_bag2"],
    },
    {
        "id": "ns_d14",
        "label": "WS NS Stations d14",
        "region": "north_sea",
        "horizon": 14,
        "expected_rows": 256,
        "absolute_score_mean_lte": 17.20,
        "absolute_score_max_lte": 18.60,
        "public_current": 17.0091,
        "families": ["direct_bag2", "hres_resid_bag2", "recent14_resid_bag2"],
    },
    {
        "id": "ecs_d1",
        "label": "WS ECS Stations d1",
        "region": "east_china_sea",
        "horizon": 1,
        "expected_rows": 224,
        "absolute_score_mean_lte": 6.55,
        "absolute_score_max_lte": 7.10,
        "public_current": 6.7709,
        "families": ["direct_bag2", "hres_resid_bag2", "log_bag2"],
    },
]

MODEL_SPECS = {
    "direct_bag2": {
        "kind": "quantile_bag",
        "target": "direct",
        "transform": "identity",
        "seed_offsets": [0, 1],
        "include_clim": False,
    },
    "hres_resid_bag2": {
        "kind": "quantile_bag",
        "target": "hres_resid",
        "transform": "identity",
        "seed_offsets": [0, 1],
        "include_clim": False,
    },
    "recent14_resid_bag2": {
        "kind": "quantile_bag",
        "target": "recent14_resid",
        "transform": "identity",
        "seed_offsets": [0, 1],
        "include_clim": False,
    },
    "log_bag2": {
        "kind": "quantile_bag",
        "target": "direct",
        "transform": "log1p",
        "seed_offsets": [0, 1],
        "include_clim": False,
    },
}

GROUP_STRATEGIES = [
    {"name": "no_bias", "group_cols": [], "shrink": 0.0},
    {"name": "station_s4", "group_cols": ["station"], "shrink": 4.0},
    {"name": "station_s8", "group_cols": ["station"], "shrink": 8.0},
    {"name": "station_s16", "group_cols": ["station"], "shrink": 16.0},
    {"name": "hour_s8", "group_cols": ["target_hour"], "shrink": 8.0},
    {"name": "station_hour_s24", "group_cols": ["station", "target_hour"], "shrink": 24.0},
]

CAL_GRID = [
    {"bias": bias, "k_lo": k_lo, "k_hi": k_hi}
    for bias in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10)
    for k_lo in (1.00, 1.10, 1.20, 1.30)
    for k_hi in (1.00, 1.10, 1.20, 1.30)
]

WIDTH_STRATEGIES = [
    {"name": "qwidth_global_q90", "group_cols": [], "shrink": 0.0, "q": 0.90},
    {"name": "qwidth_global_q92", "group_cols": [], "shrink": 0.0, "q": 0.92},
    {"name": "qwidth_global_q95", "group_cols": [], "shrink": 0.0, "q": 0.95},
    {"name": "qwidth_station_q92_s30", "group_cols": ["station"], "shrink": 30.0, "q": 0.92},
    {"name": "qwidth_station_q95_s30", "group_cols": ["station"], "shrink": 30.0, "q": 0.95},
    {"name": "qwidth_hour_q92_s20", "group_cols": ["target_hour"], "shrink": 20.0, "q": 0.92},
]

SEED_BASE = {
    "north_sea": {1: 20260605, 7: 20260705, 14: 20260805},
    "east_china_sea": {1: 20261605, 7: 20261705, 14: 20261805},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_member(zip_path: Path, member: str = "predictions.csv") -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def transform_y(y: np.ndarray, transform: str) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    if transform == "identity":
        return y
    if transform == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    raise ValueError(transform)


def inverse_y(y: np.ndarray, transform: str) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    if transform == "identity":
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(transform)


def baseline_values(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "direct":
        return np.zeros(len(df), dtype="float64")
    if target == "hres_resid":
        vals = pd.to_numeric(df["hres_speed"], errors="coerce").to_numpy(dtype="float64", copy=True)
    elif target == "recent14_resid":
        vals = pd.to_numeric(df["recent14_speed"], errors="coerce").to_numpy(dtype="float64", copy=True)
        fallback = pd.to_numeric(df["recent7_speed"], errors="coerce").to_numpy(dtype="float64", copy=True)
        vals[~np.isfinite(vals)] = fallback[~np.isfinite(vals)]
        fallback = pd.to_numeric(df["hres_speed"], errors="coerce").to_numpy(dtype="float64", copy=True)
        vals[~np.isfinite(vals)] = fallback[~np.isfinite(vals)]
    else:
        raise ValueError(target)
    vals[~np.isfinite(vals)] = 0.0
    return vals


def clean_x(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def fit_quantile_model(train: pd.DataFrame, feats: list[str], spec: dict[str, object], seed: int) -> dict[str, object]:
    ok = np.isfinite(train["y_speed"].to_numpy(dtype="float64"))
    X = clean_x(train.loc[ok], feats)
    y = train.loc[ok, "y_speed"].to_numpy(dtype="float64")
    target = str(spec["target"])
    transform = str(spec["transform"])
    if target == "direct":
        y_fit = transform_y(y, transform)
    else:
        y_fit = y - baseline_values(train.loc[ok], target)
    models = (
        CV.fit_lgbm(X, y_fit, "quantile", seed + 1, N_ESTIMATORS, 0.05),
        CV.fit_lgbm(X, y_fit, "quantile", seed + 2, N_ESTIMATORS, 0.50),
        CV.fit_lgbm(X, y_fit, "quantile", seed + 3, N_ESTIMATORS, 0.95),
    )
    return {"spec": spec, "features": feats, "models": models}


def predict_quantile_model(model: dict[str, object], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = model["spec"]
    feats = model["features"]
    m05, m50, m95 = model["models"]
    X = clean_x(df, feats)
    q05 = m05.predict(X).astype("float64")
    q50 = m50.predict(X).astype("float64")
    q95 = m95.predict(X).astype("float64")
    target = str(spec["target"])
    transform = str(spec["transform"])
    if target == "direct":
        q05 = inverse_y(q05, transform)
        q50 = inverse_y(q50, transform)
        q95 = inverse_y(q95, transform)
    else:
        base = baseline_values(df, target)
        q05 = base + q05
        q50 = base + q50
        q95 = base + q95
    lo = np.maximum(0.0, np.minimum(q05, q50))
    mid = np.maximum(0.0, q50)
    hi = np.maximum(q95, mid)
    return lo, mid, hi


def fit_bagged_model(train: pd.DataFrame, feats: list[str], target: dict[str, object], spec_name: str, val_year: int | None) -> dict[str, object]:
    spec = MODEL_SPECS[spec_name]
    base_seed = int(SEED_BASE[str(target["region"])][int(target["horizon"])])
    if val_year is not None:
        base_seed += int(val_year) * 101
    else:
        base_seed += 909_000
    stable_spec_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(spec_name)) % 1000
    models = []
    for offset in spec["seed_offsets"]:
        seed = base_seed + int(offset) * 997 + stable_spec_offset
        print(f"  fitting {target['id']} {spec_name} seed={seed}", flush=True)
        models.append(fit_quantile_model(train, feats, spec, seed))
    return {"spec_name": spec_name, "spec": spec, "features": feats, "models": models}


def predict_bagged_model(model: dict[str, object], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = [predict_quantile_model(m, df) for m in model["models"]]
    lo = np.nanmean(np.vstack([p[0] for p in preds]), axis=0)
    mid = np.nanmean(np.vstack([p[1] for p in preds]), axis=0)
    hi = np.nanmean(np.vstack([p[2] for p in preds]), axis=0)
    lo = np.maximum(0.0, np.minimum(lo, mid))
    mid = np.maximum(0.0, mid)
    hi = np.maximum(hi, mid)
    return lo, mid, hi


def load_target_training(target: dict[str, object], meta: pd.DataFrame) -> pd.DataFrame:
    region = str(target["region"])
    horizon = int(target["horizon"])
    train_base = CV.load_station_origin_rows(region, meta)
    hist = CV.make_history(CV.load_station_obs(region))
    include_clim = any(bool(MODEL_SPECS[f].get("include_clim", False)) for f in target["families"])
    df = CV.build_combo(train_base, hist, horizon, include_climatology=include_clim)
    df = df[df["y_speed"].notna()].copy()
    df["__row_id"] = np.arange(len(df), dtype="int64")
    return df.reset_index(drop=True)


def feature_list(df: pd.DataFrame) -> list[str]:
    feats = CV.numeric_features(df)
    return [f for f in feats if f != "__row_id"]


def evaluate_model_grid(
    target: dict[str, object],
    train: pd.DataFrame,
    val: pd.DataFrame,
    model: dict[str, object],
    val_year: int,
) -> list[dict[str, object]]:
    tr_lo, tr_mid, tr_hi = predict_bagged_model(model, train)
    vl_lo, vl_mid, vl_hi = predict_bagged_model(model, val)
    y_tr = train["y_speed"].to_numpy(dtype="float64")
    y_vl = val["y_speed"].to_numpy(dtype="float64")
    rows = []
    for strategy in GROUP_STRATEGIES:
        group_cols = list(strategy["group_cols"])
        bias_map, global_bias = build_bias_map(train, y_tr - tr_mid, group_cols, float(strategy["shrink"]))
        tr_bias = lookup_bias(train, bias_map, global_bias, group_cols)
        vl_bias = lookup_bias(val, bias_map, global_bias, group_cols)
        tr_lo2, tr_mid2, tr_hi2 = shift_predictions(tr_lo, tr_mid, tr_hi, tr_bias)
        vl_lo2, vl_mid2, vl_hi2 = shift_predictions(vl_lo, vl_mid, vl_hi, vl_bias)
        for cal in CAL_GRID:
            lo, _, hi = apply_cal_dict(vl_lo2, vl_mid2, vl_hi2, cal)
            rows.append(
                {
                    "target_id": str(target["id"]),
                    "label": str(target["label"]),
                    "region": str(target["region"]),
                    "horizon": int(target["horizon"]),
                    "val_year": int(val_year),
                    "mode": "model_interval",
                    "family": str(model["spec_name"]),
                    "strategy": str(strategy["name"]),
                    "group_cols": ",".join(group_cols),
                    "shrink": float(strategy["shrink"]),
                    "width_strategy": "model_interval",
                    "width_group_cols": "",
                    "width_shrink": -1.0,
                    "width_q": -1.0,
                    "cal_bias": float(cal["bias"]),
                    "cal_k_lo": float(cal["k_lo"]),
                    "cal_k_hi": float(cal["k_hi"]),
                    "global_bias": float(global_bias),
                    "score": CV.speed_winkler(y_vl, lo, hi),
                    "width": float(np.nanmean(hi - lo)),
                    "n": int(np.isfinite(y_vl).sum()),
                }
            )
        for width_strategy in WIDTH_STRATEGIES:
            width_cols = list(width_strategy["group_cols"])
            lower_map, upper_map, global_lower, global_upper = build_width_maps(
                train,
                tr_mid2,
                width_cols,
                float(width_strategy["shrink"]),
                float(width_strategy["q"]),
            )
            lower, upper = lookup_widths(val, lower_map, upper_map, global_lower, global_upper, width_cols)
            lo, _, hi = residual_width_interval(vl_mid2, lower, upper)
            rows.append(
                {
                    "target_id": str(target["id"]),
                    "label": str(target["label"]),
                    "region": str(target["region"]),
                    "horizon": int(target["horizon"]),
                    "val_year": int(val_year),
                    "mode": "residual_width",
                    "family": str(model["spec_name"]),
                    "strategy": str(strategy["name"]),
                    "group_cols": ",".join(group_cols),
                    "shrink": float(strategy["shrink"]),
                    "width_strategy": str(width_strategy["name"]),
                    "width_group_cols": ",".join(width_cols),
                    "width_shrink": float(width_strategy["shrink"]),
                    "width_q": float(width_strategy["q"]),
                    "cal_bias": 0.0,
                    "cal_k_lo": 0.0,
                    "cal_k_hi": 0.0,
                    "global_bias": float(global_bias),
                    "score": CV.speed_winkler(y_vl, lo, hi),
                    "width": float(np.nanmean(hi - lo)),
                    "n": int(np.isfinite(y_vl).sum()),
                }
            )
    return rows


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "target_id",
        "label",
        "region",
        "horizon",
        "mode",
        "family",
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
    return (
        cv.groupby(group_cols, as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            width_mean=("width", "mean"),
            global_bias_mean=("global_bias", "mean"),
            n_min=("n", "min"),
        )
        .sort_values(["target_id", "score_mean", "score_max", "width_mean"], kind="mergesort")
        .reset_index(drop=True)
    )


def annotate_and_select(target: dict[str, object], summary: pd.DataFrame) -> dict[str, object]:
    sub = summary[summary["target_id"].eq(str(target["id"]))].copy()
    if "baseline_cv_mean" in target:
        sub["baseline_cv_mean"] = float(target["baseline_cv_mean"])
        sub["baseline_cv_max"] = float(target["baseline_cv_max"])
        sub["mean_improvement"] = float(target["baseline_cv_mean"]) - sub["score_mean"].astype(float)
        sub["max_delta"] = sub["score_max"].astype(float) - float(target["baseline_cv_max"])
        passed = sub[
            sub["mean_improvement"].ge(float(target["required_mean_improvement"]))
            & sub["max_delta"].le(float(target["max_margin"]))
        ].copy()
        gate = {
            "kind": "relative_to_current_accepted_cv",
            "baseline_cv_mean": float(target["baseline_cv_mean"]),
            "baseline_cv_max": float(target["baseline_cv_max"]),
            "required_mean_improvement": float(target["required_mean_improvement"]),
            "max_margin": float(target["max_margin"]),
        }
    else:
        sub["baseline_cv_mean"] = np.nan
        sub["baseline_cv_max"] = np.nan
        sub["mean_improvement"] = np.nan
        sub["max_delta"] = np.nan
        passed = sub[
            sub["score_mean"].astype(float).le(float(target["absolute_score_mean_lte"]))
            & sub["score_max"].astype(float).le(float(target["absolute_score_max_lte"]))
        ].copy()
        gate = {
            "kind": "absolute_sanity_gate",
            "score_mean_lte": float(target["absolute_score_mean_lte"]),
            "score_max_lte": float(target["absolute_score_max_lte"]),
        }
    if len(passed):
        selected = passed.sort_values(["score_mean", "score_max", "width_mean"], kind="mergesort").iloc[0].to_dict()
        selected["gate_passed"] = True
        selected["selection_rule"] = "best_mean_among_gate_passed"
    else:
        selected = sub.sort_values(["score_mean", "score_max", "width_mean"], kind="mergesort").iloc[0].to_dict()
        selected["gate_passed"] = False
        selected["selection_rule"] = "best_mean_gate_failed"
    selected["gate"] = gate
    selected["public_current"] = float(target["public_current"])
    print(f"\nTop candidates for {target['label']}:", flush=True)
    show_cols = [
        "target_id",
        "mode",
        "family",
        "strategy",
        "width_strategy",
        "cal_bias",
        "cal_k_lo",
        "cal_k_hi",
        "score_mean",
        "score_max",
        "width_mean",
    ]
    print(sub.sort_values(["score_mean", "score_max"]).head(12)[show_cols].to_string(index=False), flush=True)
    print(f"Selected {target['id']}: {selected}", flush=True)
    return selected


def run_cv_gate(training_by_target: dict[str, pd.DataFrame]) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame]:
    rows = []
    for target in TARGETS:
        df = training_by_target[str(target["id"])]
        for val_year in (2020, 2021):
            train = df[df["time"].dt.year.lt(val_year)].copy()
            val = df[CV.anchor_mask(df, val_year)].copy()
            feats = feature_list(train)
            print(
                f"CV {target['id']} fold {val_year}: train={len(train):,} val={len(val):,} features={len(feats)}",
                flush=True,
            )
            for family in target["families"]:
                model = fit_bagged_model(train, feats, target, str(family), val_year=val_year)
                rows.extend(evaluate_model_grid(target, train, val, model, val_year))
    cv = pd.DataFrame(rows)
    summary = summarize_cv(cv)
    selected = {str(target["id"]): annotate_and_select(target, summary) for target in TARGETS}
    merged = cv.merge(
        summary.add_prefix("summary_"),
        left_on=[
            "target_id",
            "label",
            "region",
            "horizon",
            "mode",
            "family",
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
        right_on=[
            "summary_target_id",
            "summary_label",
            "summary_region",
            "summary_horizon",
            "summary_mode",
            "summary_family",
            "summary_strategy",
            "summary_group_cols",
            "summary_shrink",
            "summary_width_strategy",
            "summary_width_group_cols",
            "summary_width_shrink",
            "summary_width_q",
            "summary_cal_bias",
            "summary_cal_k_lo",
            "summary_cal_k_hi",
        ],
        how="left",
    )
    merged.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}", flush=True)
    return selected, cv, summary


def fit_final_target(target: dict[str, object], train_df: pd.DataFrame, selected: dict[str, object]) -> dict[str, object]:
    spec_name = str(selected["family"])
    feats = feature_list(train_df)
    print(f"Fitting final {target['id']} {spec_name}: rows={len(train_df):,} features={len(feats)}", flush=True)
    model = fit_bagged_model(train_df, feats, target, spec_name, val_year=None)
    tr_lo, tr_mid, tr_hi = predict_bagged_model(model, train_df)
    group_cols = [c for c in str(selected["group_cols"]).split(",") if c]
    bias_map, global_bias = build_bias_map(
        train_df,
        train_df["y_speed"].to_numpy(dtype="float64") - tr_mid,
        group_cols,
        float(selected["shrink"]),
    )
    width_cols = [c for c in str(selected["width_group_cols"]).split(",") if c]
    width_fit: dict[str, object] = {}
    if str(selected["mode"]) == "residual_width":
        train_bias = lookup_bias(train_df, bias_map, global_bias, group_cols)
        _, tr_center, _ = shift_predictions(tr_lo, tr_mid, tr_hi, train_bias)
        lower_map, upper_map, global_lower, global_upper = build_width_maps(
            train_df,
            tr_center,
            width_cols,
            float(selected["width_shrink"]),
            float(selected["width_q"]),
        )
        width_fit = {
            "lower_map": lower_map,
            "upper_map": upper_map,
            "global_lower": float(global_lower),
            "global_upper": float(global_upper),
        }
    return {
        "model": model,
        "group_cols": group_cols,
        "bias_map": bias_map,
        "global_bias": float(global_bias),
        "width_cols": width_cols,
        "width_fit": width_fit,
        "cal": {
            "bias": float(selected["cal_bias"]),
            "k_lo": float(selected["cal_k_lo"]),
            "k_hi": float(selected["cal_k_hi"]),
        },
    }


def predict_final_interval(fit: dict[str, object], selected: dict[str, object], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, mid, hi = predict_bagged_model(fit["model"], df)
    bias = lookup_bias(df, fit["bias_map"], float(fit["global_bias"]), fit["group_cols"])
    lo, mid, hi = shift_predictions(lo, mid, hi, bias)
    if str(selected["mode"]) == "residual_width":
        width_fit = fit["width_fit"]
        lower, upper = lookup_widths(
            df,
            width_fit["lower_map"],
            width_fit["upper_map"],
            float(width_fit["global_lower"]),
            float(width_fit["global_upper"]),
            fit["width_cols"],
        )
        lo, mid, hi = residual_width_interval(mid, lower, upper)
    else:
        lo, mid, hi = apply_cal_dict(lo, mid, hi, fit["cal"])
    return lo, mid, hi


def make_target_patch(target: dict[str, object], meta: pd.DataFrame, train_df: pd.DataFrame, selected: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    region = str(target["region"])
    horizon = int(target["horizon"])
    spec_name = str(selected["family"])
    include_clim = bool(MODEL_SPECS[spec_name].get("include_clim", False))
    fit = fit_final_target(target, train_df, selected)
    patch_rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(region, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(region, window))
        for hour in HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, horizon, hour, include_climatology=include_clim)
            inf["__row_id"] = -1
            lo, mid, hi = predict_final_interval(fit, selected, inf)
            for station, q05, q50, q95 in zip(inf["station"].astype(str), lo, mid, hi):
                patch_rows.append(
                    {
                        "window": int(window),
                        "region": region,
                        "station": station,
                        "horizon": horizon,
                        "hour": int(hour),
                        "q05_new": float(max(q05, 0.0)),
                        "q50_new": float(max(q50, 0.0)),
                        "q95_new": float(max(q95, q50, 0.0)),
                        "target_id": str(target["id"]),
                    }
                )
    final_meta = {
        "family": spec_name,
        "group_cols": fit["group_cols"],
        "global_bias": float(fit["global_bias"]),
        "n_bias_keys": int(len(fit["bias_map"])),
        "width_cols": fit["width_cols"],
        "mode": str(selected["mode"]),
        "cal": fit["cal"],
    }
    if fit["width_fit"]:
        final_meta["global_lower"] = float(fit["width_fit"]["global_lower"])
        final_meta["global_upper"] = float(fit["width_fit"]["global_upper"])
        final_meta["n_width_keys"] = int(len(fit["width_fit"]["lower_map"]))
    return pd.DataFrame(patch_rows), final_meta


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool = False) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def apply_patches(base: pd.DataFrame, patches: list[pd.DataFrame], selected_by_target: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    if not patches:
        raise SystemExit("No station-speed target passed its gate; no submission zip written.")
    patch = pd.concat(patches, ignore_index=True)
    key = ["window", "region", "station", "horizon", "hour"]
    merged = base.reset_index().merge(patch, on=key, how="left", validate="many_to_one")
    target = merged["type"].eq("station") & merged["q50_new"].notna()
    patch_counts: dict[str, int] = {}
    for target_cfg in TARGETS:
        tid = str(target_cfg["id"])
        if bool(selected_by_target[tid].get("gate_passed")):
            count = int((target & merged["target_id"].eq(tid)).sum())
            expected = int(target_cfg["expected_rows"])
            if count != expected:
                raise SystemExit(f"{tid}: expected {expected} patched rows, got {count}")
            patch_counts[tid] = count
    before = base.copy()
    for c in SPEED_COLS:
        merged.loc[target, c] = merged.loc[target, f"{c}_new"]
    out = (
        merged.drop(columns=["q05_new", "q50_new", "q95_new", "target_id"])
        .set_index("index")
        .sort_index()[COLS]
    )
    speed_changed = rows_changed(before, out, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, out, DIR_COLS, 1, circular=True)
    changed_allowed = target.to_numpy(dtype=bool)
    outside_speed = speed_changed & ~changed_allowed
    if int(outside_speed.sum()) != 0 or int(dir_changed.sum()) != 0:
        raise SystemExit(f"Unexpected delta: outside_speed={int(outside_speed.sum())}, dir_changed={int(dir_changed.sum())}")
    delta = {
        "patched_counts": patch_counts,
        "target_rows": int(target.sum()),
        "speed_rows_changed": int(speed_changed.sum()),
        "non_target_speed_rows_changed": int(outside_speed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
    }
    return out, delta


def write_gate_failed_manifest(selected_by_target: dict[str, dict[str, object]], summary: pd.DataFrame, cv: pd.DataFrame) -> None:
    payload = {
        "status": "gate_failed_no_submission_written",
        "selected_by_target": selected_by_target,
        "cv_rows": int(len(cv)),
        "summary_head": summary.groupby("target_id", group_keys=False).head(12).to_dict(orient="records"),
        "compliance": [
            "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Gate failed for all station-speed targets, so no submission zip was emitted.",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote gate-failed manifest {MANIFEST}", flush=True)


def write_manifest(
    final: pd.DataFrame,
    selected_by_target: dict[str, dict[str, object]],
    final_fit_meta: dict[str, dict[str, object]],
    delta: dict[str, object],
    summary: pd.DataFrame,
) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")
    manifest = {
        "submission": {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "zip_name_length": len(OUT_ZIP.name),
            "csv_size": int(OUT_CSV.stat().st_size),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_names": names,
            "internal_csv_size": int(info.file_size),
            "internal_csv_sha256": sha256_zip_member(OUT_ZIP),
            "testzip": bad,
        },
        "base_csv": {"path": str(BASE_CSV), "size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
        "selected_by_target": selected_by_target,
        "final_fit_meta": final_fit_meta,
        "delta": delta,
        "cv": {
            "summary_csv": str(SUMMARY_CSV),
            "summary_sha256": sha256(SUMMARY_CSV),
            "summary_head": summary.groupby("target_id", group_keys=False).head(12).to_dict(orient="records"),
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "notes": [
                "Training labels come only from official historical station observations.",
                "Inference history uses provided context station files for each official window.",
                "Each station-speed target is independently gated by chronological CV.",
                "Failed targets remain unchanged in the emitted submission.",
            ],
        },
        "code_hashes": {
            "build_station_speed_bag_gate_candidate.py": sha256(Path(__file__).resolve()),
            "build_station_d1_direction_bag_gate_candidate.py": sha256(ROOT / "build_station_d1_direction_bag_gate_candidate.py"),
            "build_ns_station_d1_speed_calib_gate_candidate.py": sha256(ROOT / "build_ns_station_d1_speed_calib_gate_candidate.py"),
            "station_cv_mos_analog_framework.py": sha256(ROOT / "station_cv_mos_analog_framework.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_station_d1_direction_bag_gate_e2e.ps1 first.")
    print(f"Reading current best base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    meta = CV.read_station_meta()
    training_by_target = {}
    for target in TARGETS:
        df = load_target_training(target, meta)
        training_by_target[str(target["id"])] = df
        print(f"Loaded {target['id']} training rows={len(df):,}", flush=True)

    selected_by_target, cv, summary = run_cv_gate(training_by_target)
    patches = []
    final_fit_meta = {}
    for target in TARGETS:
        tid = str(target["id"])
        selected = selected_by_target[tid]
        if bool(selected.get("gate_passed")):
            print(f"Gate passed for {tid}; building final patch", flush=True)
            patch, meta_fit = make_target_patch(target, meta, training_by_target[tid], selected)
            patches.append(patch)
            final_fit_meta[tid] = meta_fit
        else:
            print(f"Leaving {tid} unchanged; gate failed", flush=True)
    if not patches:
        write_gate_failed_manifest(selected_by_target, summary, cv)
        raise SystemExit("No station-speed target passed its gate; no submission zip written.")

    patched, delta = apply_patches(base, patches, selected_by_target)
    final = E2E.validate_final(patched)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, selected_by_target, final_fit_meta, delta, summary)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
