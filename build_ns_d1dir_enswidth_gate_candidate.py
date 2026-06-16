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
from build_station_uvres_ns_d1_direction_gate_on_stage2_candidate import (
    apply_angle_bias,
    build_angle_bias_map,
    fit_xy_models,
    lookup_angle_bias,
    predict_xy_direction,
)


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "predictions_station_uvres_ns_d1_dir_gate_on_stage2_compact.csv"
OUT_CSV = WORK / "pred_ns_d1dir_ensw_gate.csv"
OUT_ZIP = WORK / "sub_ns_d1dir_ensw_gate.zip"
SUMMARY_CSV = WORK / "cv_ns_d1dir_ensw_gate.csv"
MANIFEST = WORK / "manifest_ns_d1dir_ensw_gate.json"

SEED = 20260527
LGB_ESTIMATORS = 360
CURRENT_PUBLIC_METRIC = 186.9108
CURRENT_CV_MEAN = 189.20529004597734
CURRENT_CV_MAX = 207.48799979844796
STRICT_GATE_MEAN = 188.70
STRICT_GATE_MAX = 207.80
STRICT_GATE_IMPROVEMENT = 0.35

WIDTH_GRID = np.array([45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0], dtype="float64")
WIDTH_GRID_SAFE = np.array([55.0, 57.5, 60.0, 62.5, 65.0, 70.0, 75.0], dtype="float64")

MEMBERS = {
    "du_none": {"family": "direct_unit", "group_cols": [], "shrink": 0.0},
    "du_s8": {"family": "direct_unit", "group_cols": ["station"], "shrink": 8.0},
    "du_s20": {"family": "direct_unit", "group_cols": ["station"], "shrink": 20.0},
    "du_sh12": {"family": "direct_unit", "group_cols": ["station", "target_hour"], "shrink": 12.0},
    "du_sh32": {"family": "direct_unit", "group_cols": ["station", "target_hour"], "shrink": 32.0},
    "ur_s8": {"family": "unit_residual", "group_cols": ["station"], "shrink": 8.0},
}

CENTER_SPECS = [
    {"name": "du_s8", "members": ["du_s8"]},
    {"name": "du_s20", "members": ["du_s20"]},
    {"name": "du_sh12", "members": ["du_sh12"]},
    {"name": "du_sh32", "members": ["du_sh32"]},
    {"name": "ens_s8_s20_sh12", "members": ["du_s8", "du_s20", "du_sh12"]},
    {"name": "ens_s8_sh12_sh32", "members": ["du_s8", "du_sh12", "du_sh32"]},
    {"name": "ens_s8_s20_ur8", "members": ["du_s8", "du_s20", "ur_s8"]},
]

WIDTH_POLICIES = [
    {"name": "fixed55", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 55.0},
    {"name": "fixed57_5", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 57.5},
    {"name": "fixed58_5", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 58.5},
    {"name": "fixed60", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 60.0},
    {"name": "fixed65", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 65.0},
    {"name": "train_global", "kind": "adaptive", "group_cols": [], "shrink": 0.0},
    {"name": "station_s16", "kind": "adaptive", "group_cols": ["station"], "shrink": 16.0},
    {"name": "station_s40", "kind": "adaptive", "group_cols": ["station"], "shrink": 40.0},
    {"name": "hour_s12", "kind": "adaptive", "group_cols": ["target_hour"], "shrink": 12.0},
    {"name": "station_hour_s32", "kind": "adaptive", "group_cols": ["station", "target_hour"], "shrink": 32.0},
    {"name": "station_hour_s80", "kind": "adaptive", "group_cols": ["station", "target_hour"], "shrink": 80.0},
    {"name": "station_safe_s16", "kind": "adaptive", "group_cols": ["station"], "shrink": 16.0, "safe_grid": True},
    {"name": "station_safe_s40", "kind": "adaptive", "group_cols": ["station"], "shrink": 40.0, "safe_grid": True},
    {"name": "hour_safe_s12", "kind": "adaptive", "group_cols": ["target_hour"], "shrink": 12.0, "safe_grid": True},
    {"name": "station_hour_safe_s32", "kind": "adaptive", "group_cols": ["station", "target_hour"], "shrink": 32.0, "safe_grid": True},
    {"name": "station_hour_safe_s80", "kind": "adaptive", "group_cols": ["station", "target_hour"], "shrink": 80.0, "safe_grid": True},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["__global__"] * len(df), index=df.index)
    return df[group_cols].astype(str).agg("|".join, axis=1)


def circ_mean_matrix(preds: list[np.ndarray]) -> np.ndarray:
    arr = np.vstack([np.asarray(p, dtype="float64") % 360.0 for p in preds])
    x = np.nanmean(np.cos(np.radians(arr)), axis=0)
    y = np.nanmean(np.sin(np.radians(arr)), axis=0)
    return np.degrees(np.arctan2(y, x)) % 360.0


def direction_score_var(y: np.ndarray, center: np.ndarray, half_width: np.ndarray | float) -> float:
    y = np.asarray(y, dtype="float64") % 360.0
    center = np.asarray(center, dtype="float64") % 360.0
    hw = np.asarray(half_width, dtype="float64")
    if hw.ndim == 0:
        hw = np.full(len(y), float(hw), dtype="float64")
    ok = np.isfinite(y) & np.isfinite(center) & np.isfinite(hw)
    if not bool(ok.any()):
        return float("nan")
    y = y[ok]
    center = center[ok]
    hw = np.clip(hw[ok], 20.0, 179.9)
    lo = (center - hw) % 360.0
    hi = (center + hw) % 360.0
    width = (hi - lo) % 360.0
    inside = ((y - lo) % 360.0) <= width
    miss = np.minimum(CV.circ_abs_diff(y, lo), CV.circ_abs_diff(y, hi))
    return float(np.mean(width + 20.0 * miss * (~inside)))


def best_half_width(y: np.ndarray, center: np.ndarray, grid: np.ndarray = WIDTH_GRID) -> tuple[float, float]:
    best_score = float("inf")
    best_width = float(grid[0])
    for width in grid:
        score = direction_score_var(y, center, float(width))
        if score < best_score:
            best_score = score
            best_width = float(width)
    return best_width, best_score


def fit_width_policy(train: pd.DataFrame, center: np.ndarray, policy: dict) -> dict[str, object]:
    y = train["y_dir"].to_numpy(dtype="float64")
    if policy["kind"] == "fixed":
        return {"kind": "fixed", "value": float(policy["value"])}
    grid = WIDTH_GRID_SAFE if bool(policy.get("safe_grid", False)) else WIDTH_GRID
    global_width, global_score = best_half_width(y, center, grid)
    group_cols = list(policy["group_cols"])
    if not group_cols:
        return {"kind": "adaptive", "group_cols": [], "global_width": global_width, "width_map": {}, "global_score": global_score}
    tmp = train[group_cols].copy()
    tmp["_y"] = y
    tmp["_center"] = center
    rows = []
    for key, g in tmp.groupby(group_cols, dropna=False, sort=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        width, _ = best_half_width(g["_y"].to_numpy(dtype="float64"), g["_center"].to_numpy(dtype="float64"), grid)
        n = len(g)
        weight = n / (n + float(policy["shrink"]))
        rows.append({**{c: v for c, v in zip(group_cols, key_vals)}, "width": weight * width + (1.0 - weight) * global_width})
    width_df = pd.DataFrame(rows)
    width_map = pd.Series(width_df["width"].to_numpy(dtype="float64"), index=key_frame(width_df, group_cols))
    return {
        "kind": "adaptive",
        "group_cols": group_cols,
        "global_width": global_width,
        "global_score": global_score,
        "width_map": width_map,
    }


def predict_width(df: pd.DataFrame, policy_fit: dict[str, object]) -> np.ndarray:
    if policy_fit["kind"] == "fixed":
        return np.full(len(df), float(policy_fit["value"]), dtype="float64")
    group_cols = list(policy_fit["group_cols"])
    global_width = float(policy_fit["global_width"])
    if not group_cols:
        return np.full(len(df), global_width, dtype="float64")
    width_map = policy_fit["width_map"]
    return key_frame(df, group_cols).map(width_map).fillna(global_width).to_numpy(dtype="float64")


OLD_FAMILY_INDEX = {"unit_residual": 0, "speed_uv_residual": 1, "direct_unit": 2, "direct_speed_uv": 3}


def fit_member_models(train: pd.DataFrame, feats: list[str], val_year: int | None = None) -> dict[str, tuple[object, object]]:
    models_by_family = {}
    for family in sorted({m["family"] for m in MEMBERS.values()}):
        print(f"  fitting raw {family}", flush=True)
        if val_year is None:
            seed = SEED + 9999
        else:
            seed = SEED + int(val_year) * 100 + OLD_FAMILY_INDEX[family] * 10
        models_by_family[family] = fit_xy_models(train, feats, family, seed)
    return models_by_family


def calibrated_member_center(
    member_name: str,
    train: pd.DataFrame,
    pred_df: pd.DataFrame,
    raw_train: dict[str, np.ndarray],
    raw_pred: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.Series, float]:
    spec = MEMBERS[member_name]
    group_cols = list(spec["group_cols"])
    family = str(spec["family"])
    bias_map, global_bias = build_angle_bias_map(train, raw_train[family], group_cols, float(spec["shrink"]))
    pred = apply_angle_bias(raw_pred[family], lookup_angle_bias(pred_df, bias_map, global_bias, group_cols))
    return pred, bias_map, global_bias


def build_centers(
    train: pd.DataFrame,
    pred_df: pd.DataFrame,
    raw_train: dict[str, np.ndarray],
    raw_pred: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    member_preds = {}
    member_fits = {}
    for name in sorted({m for spec in CENTER_SPECS for m in spec["members"]}):
        pred, bias_map, global_bias = calibrated_member_center(name, train, pred_df, raw_train, raw_pred)
        member_preds[name] = pred
        member_fits[name] = {"bias_map": bias_map, "global_bias": global_bias}
    centers = {}
    for spec in CENTER_SPECS:
        centers[spec["name"]] = circ_mean_matrix([member_preds[name] for name in spec["members"]])
    return centers, member_fits


def evaluate_fold(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], val_year: int) -> tuple[list[dict], dict[str, object]]:
    print(f"CV fold {val_year}: train={len(train):,} val={len(val):,} features={len(feats)}", flush=True)
    models_by_family = fit_member_models(train, feats, val_year=val_year)
    raw_train = {family: predict_xy_direction(model, feats, train, family) for family, model in models_by_family.items()}
    raw_val = {family: predict_xy_direction(model, feats, val, family) for family, model in models_by_family.items()}
    train_centers, _ = build_centers(train, train, raw_train, raw_train)
    val_centers, _ = build_centers(train, val, raw_train, raw_val)
    y_val = val["y_dir"].to_numpy(dtype="float64")
    rows = []
    for center_spec in CENTER_SPECS:
        center_name = center_spec["name"]
        for width_policy in WIDTH_POLICIES:
            fit = fit_width_policy(train, train_centers[center_name], width_policy)
            widths = predict_width(val, fit)
            score = direction_score_var(y_val, val_centers[center_name], widths)
            rows.append(
                {
                    "val_year": val_year,
                    "center": center_name,
                    "members": ",".join(center_spec["members"]),
                    "width_policy": width_policy["name"],
                    "width_kind": width_policy["kind"],
                    "width_group_cols": ",".join(width_policy["group_cols"]),
                    "width_shrink": float(width_policy["shrink"]),
                    "score": float(score),
                    "half_width_mean": float(np.nanmean(widths)),
                    "half_width_min": float(np.nanmin(widths)),
                    "half_width_max": float(np.nanmax(widths)),
                    "train_rows": len(train),
                    "val_rows": len(val),
                }
            )
    return rows, {"models": models_by_family}


def select_candidate(cv: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = (
        cv.groupby(["center", "members", "width_policy", "width_kind", "width_group_cols", "width_shrink"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            half_width_mean=("half_width_mean", "mean"),
            half_width_min=("half_width_min", "min"),
            half_width_max=("half_width_max", "max"),
        )
        .sort_values(["score_mean", "score_max"])
        .reset_index(drop=True)
    )
    row = summary.iloc[0].to_dict()
    row["gate_passed"] = bool(
        float(row["score_mean"]) <= STRICT_GATE_MEAN
        and float(row["score_max"]) <= STRICT_GATE_MAX
        and float(row["score_mean"]) <= CURRENT_CV_MEAN - STRICT_GATE_IMPROVEMENT
    )
    row["gate_requirements"] = {
        "score_mean_lte": STRICT_GATE_MEAN,
        "score_max_lte": STRICT_GATE_MAX,
        "score_mean_better_than_current_cv_by": STRICT_GATE_IMPROVEMENT,
        "current_cv_mean": CURRENT_CV_MEAN,
        "current_cv_max": CURRENT_CV_MAX,
        "current_public_metric": CURRENT_PUBLIC_METRIC,
    }
    print("Adaptive/ensemble summary:", flush=True)
    print(summary.head(30).to_string(index=False), flush=True)
    print(f"Selected candidate: {row}", flush=True)
    return row, summary


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def write_gate_failed(selected: dict, cv: pd.DataFrame, summary: pd.DataFrame) -> None:
    MANIFEST.write_text(
        json.dumps(
            {
                "status": "gate_failed_no_submission_written",
                "selected": selected,
                "summary": summary.head(30).to_dict(orient="records"),
                "cv_rows": cv.to_dict(orient="records"),
                "compliance": [
                    "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
                    "No external datasets, no web data, and no evaluation target labels.",
                    "Gate failed, so no submission zip was emitted.",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def fit_final_components(train: pd.DataFrame, feats: list[str], selected: dict) -> tuple[dict[str, tuple[object, object]], dict[str, dict[str, object]], dict[str, object]]:
    models_by_family = fit_member_models(train, feats, val_year=None)
    raw_train = {family: predict_xy_direction(model, feats, train, family) for family, model in models_by_family.items()}
    train_centers, member_fits = build_centers(train, train, raw_train, raw_train)
    center_name = str(selected["center"])
    policy = next(p for p in WIDTH_POLICIES if p["name"] == selected["width_policy"])
    width_fit = fit_width_policy(train, train_centers[center_name], policy)
    return models_by_family, member_fits, width_fit


def predict_center_for_df(
    train: pd.DataFrame,
    pred_df: pd.DataFrame,
    feats: list[str],
    models_by_family: dict[str, tuple[object, object]],
    member_fits: dict[str, dict[str, object]],
    selected: dict,
) -> np.ndarray:
    raw_pred = {family: predict_xy_direction(model, feats, pred_df, family) for family, model in models_by_family.items()}
    center_spec = next(spec for spec in CENTER_SPECS if spec["name"] == selected["center"])
    preds = []
    for member_name in center_spec["members"]:
        spec = MEMBERS[member_name]
        group_cols = list(spec["group_cols"])
        family = str(spec["family"])
        fit = member_fits[member_name]
        preds.append(apply_angle_bias(raw_pred[family], lookup_angle_bias(pred_df, fit["bias_map"], float(fit["global_bias"]), group_cols)))
    return circ_mean_matrix(preds)


def make_patch(meta: pd.DataFrame, train: pd.DataFrame, feats: list[str], models_by_family: dict[str, tuple[object, object]], member_fits: dict[str, dict[str, object]], width_fit: dict[str, object], selected: dict) -> pd.DataFrame:
    rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            center = predict_center_for_df(train, inf, feats, models_by_family, member_fits, selected)
            widths = predict_width(inf, width_fit)
            for station, c, w in zip(inf["station"].astype(str), center, widths):
                rows.append(
                    {
                        "window": window,
                        "region": REGION,
                        "station": station,
                        "horizon": HORIZON,
                        "hour": int(hour),
                        "dir_05_new": float((c - w) % 360.0),
                        "dir_50_new": float(c % 360.0),
                        "dir_95_new": float((c + w) % 360.0),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    out = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
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
        rows, _ = evaluate_fold(train, val, feats, val_year)
        cv_rows.extend(rows)
    cv = pd.DataFrame(cv_rows)
    selected, summary = select_candidate(cv)
    cv.merge(summary.add_prefix("summary_"), left_on=["center", "members", "width_policy", "width_kind", "width_group_cols", "width_shrink"], right_on=["summary_center", "summary_members", "summary_width_policy", "summary_width_kind", "summary_width_group_cols", "summary_width_shrink"], how="left").to_csv(SUMMARY_CSV, index=False)
    if not bool(selected["gate_passed"]):
        write_gate_failed(selected, cv, summary)
        raise SystemExit("Strict adaptive/ensemble gate failed; no submission zip written.")

    train_df = df.copy()
    feats = CV.numeric_features(train_df)
    models_by_family, member_fits, width_fit = fit_final_components(train_df, feats, selected)
    patch = make_patch(meta, train_df, feats, models_by_family, member_fits, width_fit, selected)
    merged = out.reset_index().merge(patch, on=["window", "region", "station", "horizon", "hour"], how="left", validate="many_to_one")
    station_mask = merged["type"].eq("station") & merged["region"].eq(REGION) & merged["horizon"].eq(HORIZON) & merged["dir_50_new"].notna()
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
            "csv_size": int(OUT_CSV.stat().st_size),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "csv_sha256": sha256(OUT_CSV),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_size": int(info.file_size),
            "internal_names": names,
            "testzip": bad,
        },
        "base_csv": {"path": str(BASE_CSV), "size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
        "cv": {
            "summary_csv": str(SUMMARY_CSV),
            "selected": selected,
            "summary_head": summary.head(30).to_dict(orient="records"),
            "width_fit": {
                "kind": width_fit["kind"],
                "global_width": float(width_fit.get("global_width", width_fit.get("value", np.nan))),
                "group_cols": list(width_fit.get("group_cols", [])),
            },
        },
        "delta": {
            "target": "Dir NS Stations d1",
            "patched_rows": changed,
            "speed_rows_changed": speed_changed,
            "direction_rows_changed": dir_changed,
        },
        "compliance": [
            "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Training labels come only from official historical station observations.",
            "Inference history uses provided context station files for each window and only past/context values.",
            "Strict CV gate passed before emitting the submission zip.",
        ],
        "code_hashes": {
            "build_ns_d1dir_enswidth_gate_candidate.py": sha256(Path(__file__)),
            "station_cv_mos_analog_framework.py": sha256(Path("station_cv_mos_analog_framework.py")),
            "build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py": sha256(Path("build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py")),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; names={names}; testzip={bad}", flush=True)
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
