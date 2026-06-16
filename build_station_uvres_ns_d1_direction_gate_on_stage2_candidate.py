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
    SPEED_COLS,
    load_inference_origin_rows,
    load_station_obs_with_context,
    validate_submission,
)


WORK = Path("runs/v6_pressure_speed")
DATA = WORK / "phase1_dataset"
BASE_CSV = WORK / "predictions_station_lgbm_ns_d1_speed_stage2_on_analog_compact.csv"
OUT_CSV = WORK / "predictions_station_uvres_ns_d1_dir_gate_on_stage2_compact.csv"
OUT_ZIP = WORK / "submission_station_uvres_ns_d1_dir_gate_on_stage2_compact.zip"
SUMMARY_CSV = WORK / "station_uvres_ns_d1_dir_gate_cv_summary.csv"
MANIFEST = WORK / "station_uvres_ns_d1_dir_gate_manifest.json"

SEED = 20260527
LGB_ESTIMATORS = 360
PUBLIC_CURRENT_METRIC = 210.9296
STRICT_GATE_MEAN = 208.0
STRICT_GATE_MAX = 245.0
STRICT_GATE_PUBLIC_MARGIN = 2.0

MODEL_FAMILIES = ("unit_residual", "speed_uv_residual", "direct_unit", "direct_speed_uv")
CALIBRATIONS = [
    {"name": "none", "group_cols": [], "shrink": 0.0},
    {"name": "station_s8", "group_cols": ["station"], "shrink": 8.0},
    {"name": "station_s20", "group_cols": ["station"], "shrink": 20.0},
    {"name": "hour_s8", "group_cols": ["target_hour"], "shrink": 8.0},
    {"name": "station_hour_s12", "group_cols": ["station", "target_hour"], "shrink": 12.0},
    {"name": "station_hour_s32", "group_cols": ["station", "target_hour"], "shrink": 32.0},
]
WIDTH_OPTIONS = [
    {"name": "train_m0", "kind": "train", "value": 0.0},
    {"name": "train_m5", "kind": "train", "value": 5.0},
    {"name": "train_m10", "kind": "train", "value": 10.0},
    {"name": "train_m15", "kind": "train", "value": 15.0},
    {"name": "fixed55", "kind": "fixed", "value": 55.0},
    {"name": "fixed60", "kind": "fixed", "value": 60.0},
    {"name": "fixed65", "kind": "fixed", "value": 65.0},
    {"name": "fixed70", "kind": "fixed", "value": 70.0},
    {"name": "fixed75", "kind": "fixed", "value": 75.0},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def angle_xy(direction: np.ndarray, speed: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    d = np.radians(np.asarray(direction, dtype="float64") % 360.0)
    s = np.asarray(speed, dtype="float64")
    return s * np.cos(d), s * np.sin(d)


def xy_angle(x: np.ndarray, y: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    out = np.degrees(np.arctan2(y, x)) % 360.0
    bad = ~np.isfinite(out) | ((np.abs(x) + np.abs(y)) < 1e-9)
    if fallback is not None and bool(bad.any()):
        out[bad] = np.asarray(fallback, dtype="float64")[bad] % 360.0
    return out


def key_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["__global__"] * len(df), index=df.index)
    return df[group_cols].astype(str).agg("|".join, axis=1)


def circular_bias_deg(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((np.asarray(actual, dtype="float64") - np.asarray(pred, dtype="float64") + 180.0) % 360.0) - 180.0


def circular_mean_residual(resid_deg: np.ndarray) -> float:
    resid = np.asarray(resid_deg, dtype="float64")
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0:
        return 0.0
    return float(np.degrees(np.arctan2(np.sin(np.radians(resid)).mean(), np.cos(np.radians(resid)).mean())))


def build_angle_bias_map(train: pd.DataFrame, pred: np.ndarray, group_cols: list[str], shrink: float) -> tuple[pd.Series, float]:
    y = train["y_dir"].to_numpy(dtype="float64") % 360.0
    resid = circular_bias_deg(y, pred)
    ok = np.isfinite(resid)
    global_bias = circular_mean_residual(resid[ok])
    if not group_cols:
        return pd.Series(dtype="float64"), global_bias
    tmp = train.loc[ok, group_cols].copy()
    tmp["resid"] = resid[ok]
    rows = []
    for key, g in tmp.groupby(group_cols, dropna=False, sort=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        raw = circular_mean_residual(g["resid"].to_numpy(dtype="float64"))
        n = len(g)
        weight = n / (n + shrink)
        # Shrink on the unit circle by blending residual angles.
        x = weight * np.cos(np.radians(raw)) + (1.0 - weight) * np.cos(np.radians(global_bias))
        yv = weight * np.sin(np.radians(raw)) + (1.0 - weight) * np.sin(np.radians(global_bias))
        bias = float(np.degrees(np.arctan2(yv, x)))
        rows.append({**{c: v for c, v in zip(group_cols, key_vals)}, "bias": bias})
    bias_df = pd.DataFrame(rows)
    return pd.Series(bias_df["bias"].to_numpy(dtype="float64"), index=key_frame(bias_df, group_cols)), global_bias


def lookup_angle_bias(df: pd.DataFrame, bias_map: pd.Series, global_bias: float, group_cols: list[str]) -> np.ndarray:
    if not group_cols:
        return np.full(len(df), global_bias, dtype="float64")
    return key_frame(df, group_cols).map(bias_map).fillna(global_bias).to_numpy(dtype="float64")


def apply_angle_bias(pred: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.asarray(pred, dtype="float64") + np.asarray(bias, dtype="float64")) % 360.0


def fit_xy_models(train: pd.DataFrame, feats: list[str], family: str, seed: int) -> tuple[object, object]:
    ok = (
        np.isfinite(train["y_dir"].to_numpy(dtype="float64"))
        & np.isfinite(train["hres_dir"].to_numpy(dtype="float64"))
        & np.isfinite(train["hres_speed"].to_numpy(dtype="float64"))
    )
    if "speed" in family:
        ok &= np.isfinite(train["y_speed"].to_numpy(dtype="float64"))
    X = train.loc[ok, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    y_dir = train.loc[ok, "y_dir"].to_numpy(dtype="float64") % 360.0
    h_dir = train.loc[ok, "hres_dir"].to_numpy(dtype="float64") % 360.0
    h_speed = np.maximum(0.0, train.loc[ok, "hres_speed"].to_numpy(dtype="float64"))
    if family in ("unit_residual", "direct_unit"):
        target_x, target_y = angle_xy(y_dir, 1.0)
        h_x, h_y = angle_xy(h_dir, 1.0)
    elif family in ("speed_uv_residual", "direct_speed_uv"):
        y_speed = np.maximum(0.0, train.loc[ok, "y_speed"].to_numpy(dtype="float64"))
        target_x, target_y = angle_xy(y_dir, y_speed)
        h_x, h_y = angle_xy(h_dir, h_speed)
    else:
        raise ValueError(family)
    if family.endswith("residual"):
        target_x = target_x - h_x
        target_y = target_y - h_y
    mx = CV.fit_lgbm(X, target_x, "regression", seed + 1, LGB_ESTIMATORS)
    my = CV.fit_lgbm(X, target_y, "regression", seed + 2, LGB_ESTIMATORS)
    return mx, my


def predict_xy_direction(models: tuple[object, object], feats: list[str], df: pd.DataFrame, family: str) -> np.ndarray:
    mx, my = models
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    px = mx.predict(X).astype("float64")
    py = my.predict(X).astype("float64")
    h_dir = df["hres_dir"].to_numpy(dtype="float64") % 360.0
    h_speed = np.maximum(0.0, df["hres_speed"].to_numpy(dtype="float64"))
    if family == "unit_residual":
        hx, hy = angle_xy(h_dir, 1.0)
        px = hx + px
        py = hy + py
    elif family == "speed_uv_residual":
        hx, hy = angle_xy(h_dir, h_speed)
        px = hx + px
        py = hy + py
    elif family in ("direct_unit", "direct_speed_uv"):
        pass
    else:
        raise ValueError(family)
    return xy_angle(px, py, fallback=h_dir)


def train_selected_width(train: pd.DataFrame, pred: np.ndarray, margin: float) -> float:
    y = train["y_dir"].to_numpy(dtype="float64") % 360.0
    _, width = CV.best_direction_interval(y, pred)
    width = float(width) + float(margin)
    return float(min(max(width, 20.0), 179.9))


def resolve_width(train: pd.DataFrame, pred: np.ndarray, option: dict) -> float:
    if option["kind"] == "fixed":
        return float(option["value"])
    if option["kind"] == "train":
        return train_selected_width(train, pred, float(option["value"]))
    raise ValueError(option)


def evaluate_fold(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], val_year: int) -> list[dict]:
    rows = []
    for family in MODEL_FAMILIES:
        print(f"  fitting {family} for {val_year}", flush=True)
        models = fit_xy_models(train, feats, family, SEED + val_year * 100 + MODEL_FAMILIES.index(family) * 10)
        pred_train_raw = predict_xy_direction(models, feats, train, family)
        pred_val_raw = predict_xy_direction(models, feats, val, family)
        for cal in CALIBRATIONS:
            group_cols = list(cal["group_cols"])
            bias_map, global_bias = build_angle_bias_map(train, pred_train_raw, group_cols, float(cal["shrink"]))
            pred_train = apply_angle_bias(pred_train_raw, lookup_angle_bias(train, bias_map, global_bias, group_cols))
            pred_val = apply_angle_bias(pred_val_raw, lookup_angle_bias(val, bias_map, global_bias, group_cols))
            for width_option in WIDTH_OPTIONS:
                width = resolve_width(train, pred_train, width_option)
                score = CV.direction_score(val["y_dir"].to_numpy(dtype="float64"), pred_val, width)
                rows.append(
                    {
                        "val_year": val_year,
                        "family": family,
                        "calibration": cal["name"],
                        "group_cols": ",".join(group_cols),
                        "shrink": cal["shrink"],
                        "width_option": width_option["name"],
                        "width_kind": width_option["kind"],
                        "width_value": width_option["value"],
                        "half_width": width,
                        "score": score,
                        "train_rows": len(train),
                        "val_rows": len(val),
                        "global_bias": global_bias,
                    }
                )
    return rows


def select_candidate(cv: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = (
        cv.groupby(["family", "calibration", "group_cols", "shrink", "width_option", "width_kind", "width_value"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            width_mean=("half_width", "mean"),
            global_bias_mean=("global_bias", "mean"),
        )
        .sort_values(["score_mean", "score_max"])
        .reset_index(drop=True)
    )
    row = summary.iloc[0].to_dict()
    gate = (
        float(row["score_mean"]) <= STRICT_GATE_MEAN
        and float(row["score_max"]) <= STRICT_GATE_MAX
        and float(row["score_mean"]) <= PUBLIC_CURRENT_METRIC - STRICT_GATE_PUBLIC_MARGIN
    )
    row["gate_passed"] = bool(gate)
    row["gate_requirements"] = {
        "score_mean_lte": STRICT_GATE_MEAN,
        "score_max_lte": STRICT_GATE_MAX,
        "score_mean_at_least_better_than_public_by": STRICT_GATE_PUBLIC_MARGIN,
        "public_current_metric": PUBLIC_CURRENT_METRIC,
    }
    print("Candidate summary:", flush=True)
    print(summary.head(20).to_string(index=False), flush=True)
    print(f"Selected candidate: {row}", flush=True)
    return row, summary


def fit_final_bias_and_width(
    train: pd.DataFrame,
    feats: list[str],
    family: str,
    calibration: str,
    group_cols_raw: str,
    shrink: float,
    width_kind: str,
    width_value: float,
) -> tuple[tuple[object, object], pd.Series, float, float]:
    models = fit_xy_models(train, feats, family, SEED + 9999)
    pred_train_raw = predict_xy_direction(models, feats, train, family)
    group_cols = [x for x in str(group_cols_raw).split(",") if x]
    bias_map, global_bias = build_angle_bias_map(train, pred_train_raw, group_cols, float(shrink))
    pred_train = apply_angle_bias(pred_train_raw, lookup_angle_bias(train, bias_map, global_bias, group_cols))
    width = resolve_width(train, pred_train, {"kind": width_kind, "value": float(width_value)})
    return models, bias_map, global_bias, width


def predict_inference_patch(meta: pd.DataFrame, models: tuple[object, object], feats: list[str], selected: dict, bias_map: pd.Series, global_bias: float, width: float) -> pd.DataFrame:
    group_cols = [x for x in str(selected["group_cols"]).split(",") if x]
    family = str(selected["family"])
    rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            pred = predict_xy_direction(models, feats, inf, family)
            pred = apply_angle_bias(pred, lookup_angle_bias(inf, bias_map, global_bias, group_cols))
            for station, center in zip(inf["station"].astype(str), pred):
                rows.append(
                    {
                        "window": window,
                        "region": REGION,
                        "station": station,
                        "horizon": HORIZON,
                        "hour": int(hour),
                        "dir_05_new": float((center - width) % 360.0),
                        "dir_50_new": float(center % 360.0),
                        "dir_95_new": float((center + width) % 360.0),
                    }
                )
    return pd.DataFrame(rows)


def write_no_submit_manifest(selected: dict, cv: pd.DataFrame, summary: pd.DataFrame) -> None:
    payload = {
        "status": "gate_failed_no_submission_written",
        "selected_candidate": selected,
        "summary_head": summary.head(20).to_dict(orient="records"),
        "cv_rows": cv.to_dict(orient="records"),
        "compliance": [
            "Uses only files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Gate failed, so no model-patched submission zip was emitted by this guarded builder.",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote gate-failed manifest {MANIFEST}", flush=True)


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
    hist = CV.make_history(CV.load_station_obs(REGION))
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    df = df[df["y_dir"].notna() & df["hres_dir"].notna() & df["hres_speed"].notna()].copy()
    cv_rows = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year)].copy()
        val = df[CV.anchor_mask(df, val_year)].copy()
        feats = CV.numeric_features(train)
        print(f"CV fold {val_year}: train={len(train):,} val={len(val):,} features={len(feats)}", flush=True)
        cv_rows.extend(evaluate_fold(train, val, feats, val_year))
    cv = pd.DataFrame(cv_rows)
    selected, summary = select_candidate(cv)
    cv_out = cv.merge(
        summary.add_prefix("summary_"),
        left_on=["family", "calibration", "group_cols", "shrink", "width_option", "width_kind", "width_value"],
        right_on=[
            "summary_family",
            "summary_calibration",
            "summary_group_cols",
            "summary_shrink",
            "summary_width_option",
            "summary_width_kind",
            "summary_width_value",
        ],
        how="left",
    )
    cv_out.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}", flush=True)
    if not bool(selected["gate_passed"]):
        write_no_submit_manifest(selected, cv, summary)
        raise SystemExit("Strict CV gate failed; no submission zip written.")

    train_df = df.copy()
    feats = CV.numeric_features(train_df)
    models, bias_map, global_bias, width = fit_final_bias_and_width(
        train_df,
        feats,
        str(selected["family"]),
        str(selected["calibration"]),
        str(selected["group_cols"]),
        float(selected["shrink"]),
        str(selected["width_kind"]),
        float(selected["width_value"]),
    )
    print(f"Final model width={width:.1f} global_bias={global_bias:.3f}", flush=True)
    patch = predict_inference_patch(meta, models, feats, selected, bias_map, global_bias, width)
    key = ["window", "region", "station", "horizon", "hour"]
    merged = out.reset_index().merge(patch, on=key, how="left", validate="many_to_one")
    station_mask = (
        merged["type"].eq("station")
        & merged["region"].eq(REGION)
        & merged["horizon"].eq(HORIZON)
        & merged["dir_50_new"].notna()
    )
    changed = int(station_mask.sum())
    if changed != EXPECTED_PATCH_ROWS:
        raise SystemExit(f"expected {EXPECTED_PATCH_ROWS} NS station d1 direction rows, got {changed}")
    before_speed = out[SPEED_COLS].round(2).copy()
    for c in DIR_COLS:
        merged.loc[station_mask, c] = merged.loc[station_mask, f"{c}_new"]
    out2 = merged.drop(columns=["dir_05_new", "dir_50_new", "dir_95_new"]).set_index("index").sort_index()[COLS]
    speed_changed = int((before_speed.to_numpy() != out2[SPEED_COLS].round(2).to_numpy()).any(axis=1).sum())
    dir_changed = int((out[DIR_COLS].round(1).to_numpy() != out2[DIR_COLS].round(1).to_numpy()).any(axis=1).sum())
    print(f"Patched rows={changed}; speed_changed={speed_changed}; direction_changed={dir_changed}", flush=True)
    if speed_changed != 0 or dir_changed != EXPECTED_PATCH_ROWS:
        raise SystemExit("unexpected delta outside NS station d1 direction")
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
            "csv_size": OUT_CSV.stat().st_size,
            "zip_size": OUT_ZIP.stat().st_size,
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_size": info.file_size,
            "internal_names": names,
            "testzip": bad,
        },
        "base_csv": {"path": str(BASE_CSV), "size": BASE_CSV.stat().st_size, "sha256": sha256(BASE_CSV)},
        "cv_summary": {
            "path": str(SUMMARY_CSV),
            "size": SUMMARY_CSV.stat().st_size,
            "sha256": sha256(SUMMARY_CSV),
            "selected_candidate": selected,
            "summary_head": summary.head(20).to_dict(orient="records"),
            "final_half_width": width,
            "final_global_bias": global_bias,
        },
        "delta": {
            "target": "Dir NS Stations d1",
            "patched_rows": changed,
            "speed_rows_changed": speed_changed,
            "direction_rows_changed": dir_changed,
        },
        "compliance": [
            "Uses only files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Training labels come only from official historical station observations.",
            "Inference history uses provided context station files for each window and only past/context values.",
            "Strict CV gate passed before emitting the submission zip.",
        ],
        "code_hashes": {
            "build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py": sha256(Path(__file__)),
            "station_cv_mos_analog_framework.py": sha256(Path("station_cv_mos_analog_framework.py")),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; uncompressed={info.file_size:,}; names={names}; testzip={bad}", flush=True)
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
