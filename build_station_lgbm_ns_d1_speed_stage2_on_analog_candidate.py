from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import station_cv_mos_analog_framework as CV
from build_station_lgbm_ns_d1_speed_on_analog_candidate import (
    BASE_CSV,
    COLS,
    DIR_COLS,
    EXPECTED_PATCH_ROWS,
    HORIZON,
    LGB_ESTIMATORS,
    REGION,
    SEED,
    SPEED_COLS,
    WORK,
    apply_calibration,
    fit_speed_models,
    load_inference_origin_rows,
    load_station_obs_with_context,
    optimize_speed_calibration,
    predict_speed,
    sha256,
    validate_submission,
)


OUT_CSV = WORK / "predictions_station_lgbm_ns_d1_speed_stage2_on_analog_compact.csv"
OUT_ZIP = WORK / "submission_station_lgbm_ns_d1_speed_stage2_on_analog_compact.zip"
SUMMARY_CSV = WORK / "station_lgbm_ns_d1_speed_stage2_on_analog_cv_summary.csv"
MANIFEST = WORK / "station_lgbm_ns_d1_speed_stage2_on_analog_manifest.json"


STAGE2_STRATEGIES = [
    {"name": "no_bias", "group_cols": [], "shrink": 0.0, "width_scale": 1.0},
    {"name": "no_bias_w105", "group_cols": [], "shrink": 0.0, "width_scale": 1.05},
    {"name": "no_bias_w110", "group_cols": [], "shrink": 0.0, "width_scale": 1.10},
    {"name": "station_s8", "group_cols": ["station"], "shrink": 8.0, "width_scale": 1.0},
    {"name": "station_s8_w105", "group_cols": ["station"], "shrink": 8.0, "width_scale": 1.05},
    {"name": "station_s8_w110", "group_cols": ["station"], "shrink": 8.0, "width_scale": 1.10},
    {"name": "station_s20", "group_cols": ["station"], "shrink": 20.0, "width_scale": 1.0},
    {"name": "hour_s8", "group_cols": ["target_hour"], "shrink": 8.0, "width_scale": 1.0},
    {"name": "station_hour_s12", "group_cols": ["station", "target_hour"], "shrink": 12.0, "width_scale": 1.0},
    {"name": "station_hour_s12_w105", "group_cols": ["station", "target_hour"], "shrink": 12.0, "width_scale": 1.05},
    {"name": "station_hour_s32", "group_cols": ["station", "target_hour"], "shrink": 32.0, "width_scale": 1.0},
]

FIRST_BRANCH_CALIBRATION = {"bias": -0.15, "k_lo": 1.2, "k_hi": 1.2}


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


def scale_interval(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, width_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width_scale = float(width_scale)
    if abs(width_scale - 1.0) < 1e-12:
        return lo, mid, hi
    lower = np.maximum(np.asarray(mid, dtype="float64") - np.asarray(lo, dtype="float64"), 0.0)
    upper = np.maximum(np.asarray(hi, dtype="float64") - np.asarray(mid, dtype="float64"), 0.0)
    lo2 = np.maximum(0.0, np.asarray(mid, dtype="float64") - width_scale * lower)
    hi2 = np.asarray(mid, dtype="float64") + width_scale * upper
    return lo2, mid, np.maximum(hi2, mid)


def evaluate_strategy(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feats: list[str],
    models: tuple[object, object, object],
    strategy: dict,
) -> dict:
    tr_lo, tr_mid, tr_hi = predict_speed(models, feats, train)
    vl_lo, vl_mid, vl_hi = predict_speed(models, feats, val)
    y_tr = train["y_speed"].to_numpy(dtype="float64")
    y_vl = val["y_speed"].to_numpy(dtype="float64")
    group_cols = list(strategy["group_cols"])
    bias_map, global_bias = build_bias_map(train, y_tr - tr_mid, group_cols, float(strategy["shrink"]))
    tr_bias = lookup_bias(train, bias_map, global_bias, group_cols)
    vl_bias = lookup_bias(val, bias_map, global_bias, group_cols)
    tr_lo, tr_mid, tr_hi = shift_predictions(tr_lo, tr_mid, tr_hi, tr_bias)
    vl_lo, vl_mid, vl_hi = shift_predictions(vl_lo, vl_mid, vl_hi, vl_bias)
    cal = dict(FIRST_BRANCH_CALIBRATION)
    clo, cmid, chi = apply_calibration(vl_lo, vl_mid, vl_hi, cal)
    clo, cmid, chi = scale_interval(clo, cmid, chi, float(strategy["width_scale"]))
    score = CV.speed_winkler(y_vl, clo, chi)
    return {
        "score": float(score),
        "width": float(np.nanmean(chi - clo)),
        "bias_global": float(global_bias),
        "cal_bias": float(cal["bias"]),
        "cal_k_lo": float(cal["k_lo"]),
        "cal_k_hi": float(cal["k_hi"]),
        "width_scale": float(strategy["width_scale"]),
    }


def calibrate_stage2(train_base: pd.DataFrame, hist: CV.StationHistory) -> tuple[dict, pd.DataFrame]:
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    rows = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year) & df["y_speed"].notna()].copy()
        val = df[CV.anchor_mask(df, val_year) & df["y_speed"].notna()].copy()
        feats = CV.numeric_features(train)
        print(f"Fitting fold {val_year}: train={len(train):,} val={len(val):,}", flush=True)
        models = fit_speed_models(train, feats, SEED + val_year)
        for strategy in STAGE2_STRATEGIES:
            got = evaluate_strategy(train, val, feats, models, strategy)
            row = {
                "val_year": val_year,
                "strategy": strategy["name"],
                "group_cols": ",".join(strategy["group_cols"]),
                "shrink": strategy["shrink"],
                "width_scale_selected": strategy["width_scale"],
                "train_rows": len(train),
                "val_rows": len(val),
            }
            row.update(got)
            rows.append(row)
            print(
                f"CV {val_year} {strategy['name']}: score={got['score']:.4f} "
                f"width={got['width']:.3f} global_bias={got['bias_global']:.3f} "
                f"scale={got['width_scale']:.2f} "
                f"cal=({got['cal_bias']:.2f},{got['cal_k_lo']:.2f},{got['cal_k_hi']:.2f})",
                flush=True,
            )
    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby("strategy", as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            width_mean=("width", "mean"),
            global_bias_mean=("bias_global", "mean"),
            selected_width_scale=("width_scale", "mean"),
        )
        .sort_values(["score_mean", "score_max"])
        .reset_index(drop=True)
    )
    selected_name = str(summary.iloc[0]["strategy"])
    selected = next(s for s in STAGE2_STRATEGIES if s["name"] == selected_name).copy()
    selected["score_mean"] = float(summary.iloc[0]["score_mean"])
    selected["score_max"] = float(summary.iloc[0]["score_max"])
    selected["width_mean"] = float(summary.iloc[0]["width_mean"])
    selected["global_bias_mean"] = float(summary.iloc[0]["global_bias_mean"])
    selected["selected_width_scale"] = float(summary.iloc[0]["selected_width_scale"])
    cv = cv.merge(summary.add_prefix("strategy_"), left_on="strategy", right_on="strategy_strategy", how="left")
    print("Stage2 strategy summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Selected strategy: {selected}", flush=True)
    return selected, cv


def fit_final_stage2(
    train_df: pd.DataFrame,
    feats: list[str],
    models: tuple[object, object, object],
    selected: dict,
) -> tuple[pd.Series, float, dict[str, float]]:
    lo, mid, hi = predict_speed(models, feats, train_df)
    y = train_df["y_speed"].to_numpy(dtype="float64")
    group_cols = list(selected["group_cols"])
    bias_map, global_bias = build_bias_map(train_df, y - mid, group_cols, float(selected["shrink"]))
    bias = lookup_bias(train_df, bias_map, global_bias, group_cols)
    lo, mid, hi = shift_predictions(lo, mid, hi, bias)
    cal = dict(FIRST_BRANCH_CALIBRATION)
    return bias_map, global_bias, cal


def make_patch(meta: pd.DataFrame, models: tuple[object, object, object], feats: list[str], selected: dict, bias_map: pd.Series, global_bias: float, cal: dict[str, float]) -> pd.DataFrame:
    patch_rows = []
    group_cols = list(selected["group_cols"])
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            lo, mid, hi = predict_speed(models, feats, inf)
            bias = lookup_bias(inf, bias_map, global_bias, group_cols)
            lo, mid, hi = shift_predictions(lo, mid, hi, bias)
            lo, mid, hi = apply_calibration(lo, mid, hi, cal)
            lo, mid, hi = scale_interval(lo, mid, hi, float(selected["width_scale"]))
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
    return pd.DataFrame(patch_rows)


def main() -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    out = pd.read_csv(BASE_CSV, low_memory=False)[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")

    meta = CV.read_station_meta()
    train_base = CV.load_station_origin_rows(REGION, meta)
    train_hist = CV.make_history(CV.load_station_obs(REGION))
    selected, cv_summary = calibrate_stage2(train_base, train_hist)
    cv_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}", flush=True)

    train_df = CV.build_combo(train_base, train_hist, HORIZON, include_climatology=False)
    train_df = train_df[train_df["y_speed"].notna()].copy()
    feats = CV.numeric_features(train_df)
    print(f"Fitting final stage2 models: rows={len(train_df):,} features={len(feats)}", flush=True)
    models = fit_speed_models(train_df, feats, SEED + 999)
    bias_map, global_bias, cal = fit_final_stage2(train_df, feats, models, selected)
    print(f"Final stage2 global_bias={global_bias:.4f}; calibration={cal}", flush=True)

    patch = make_patch(meta, models, feats, selected, bias_map, global_bias, cal)
    key = ["window", "region", "station", "horizon", "hour"]
    merged = out.reset_index().merge(patch, on=key, how="left", validate="many_to_one")
    station_mask = (
        merged["type"].eq("station")
        & merged["region"].eq(REGION)
        & merged["horizon"].eq(HORIZON)
        & merged["q50_new"].notna()
    )
    changed = int(station_mask.sum())
    if changed != EXPECTED_PATCH_ROWS:
        raise SystemExit(f"expected {EXPECTED_PATCH_ROWS} NS station d1 speed rows, got {changed}")
    before_dir = out[DIR_COLS].round(1).copy()
    for c in SPEED_COLS:
        merged.loc[station_mask, c] = merged.loc[station_mask, f"{c}_new"]
    out2 = merged.drop(columns=["q05_new", "q50_new", "q95_new"]).set_index("index").sort_index()[COLS]
    speed_changed = int((out[SPEED_COLS].round(2).to_numpy() != out2[SPEED_COLS].round(2).to_numpy()).any(axis=1).sum())
    dir_changed = int((before_dir.to_numpy() != out2[DIR_COLS].round(1).to_numpy()).any(axis=1).sum())
    print(f"Patched rows={changed}; speed_changed={speed_changed}; direction_changed={dir_changed}", flush=True)
    if speed_changed != EXPECTED_PATCH_ROWS or dir_changed != 0:
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
    print(
        f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; "
        f"uncompressed={info.file_size:,}; names={names}; testzip={bad}",
        flush=True,
    )
    manifest = {
        "submission": {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "csv_size": OUT_CSV.stat().st_size,
            "zip_size": OUT_ZIP.stat().st_size,
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_size": info.file_size,
            "internal_names": names,
            "testzip": bad,
        },
        "base_csv": {
            "path": str(BASE_CSV),
            "size": BASE_CSV.stat().st_size,
            "sha256": sha256(BASE_CSV),
        },
        "cv_summary": {
            "path": str(SUMMARY_CSV),
            "size": SUMMARY_CSV.stat().st_size,
            "sha256": sha256(SUMMARY_CSV),
            "selected_strategy": selected,
            "final_global_bias": global_bias,
            "final_calibration": cal,
            "val_rows": cv_summary.to_dict(orient="records"),
        },
        "delta": {
            "target": "WS NS Stations d1",
            "patched_rows": changed,
            "speed_rows_changed": speed_changed,
            "direction_rows_changed": dir_changed,
        },
        "compliance": [
            "Uses only files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Training labels come only from official historical station observations.",
            "Inference history uses provided context station files for each window and only past/context values.",
            "Second-stage residual bias is fitted from historical training residuals only.",
            "Coverage calibration is fixed to the submitted first-branch calibration.",
            "The base submission is regenerated by the existing analog MOS end-to-end branch.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
