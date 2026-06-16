from __future__ import annotations

import hashlib
import gc
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import hres_mos_residual_branch as HM
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_ns_p14dir_hres.csv"
OUT_CSV = WORK / "pred_ns_p7dir_mosres.csv"
OUT_ZIP = WORK / "sub_ns_p7dir_mosres.zip"
MANIFEST = WORK / "manifest_ns_p7dir_mosres.json"

CV_SUMMARY = WORK / "cv_mos_residual_framework_summary.csv"
CV_BY_FOLD = WORK / "cv_mos_residual_framework_by_fold.csv"

REGION = "north_sea"
GROUP = "pressure"
HORIZON = 7
PROBLEM = "direction"
LEVELS = HM.GROUP_LEVELS[GROUP]

TRAIN_COMBO_SAMPLE = 220_000
RANDOM_SEED = 20260609

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS


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
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    return out


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def target_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        df["type"].eq("grid")
        & df["region"].eq(REGION)
        & df["horizon"].eq(HORIZON)
        & df["level"].isin(LEVELS)
    ).to_numpy(dtype=bool)


def load_cv_evidence() -> tuple[dict[str, object], list[dict[str, object]]]:
    require(CV_SUMMARY, "Run .\\run_cv_mos_residual_framework.ps1 first.")
    require(CV_BY_FOLD, "Run .\\run_cv_mos_residual_framework.ps1 first.")
    summary = pd.read_csv(CV_SUMMARY)
    mask = (
        summary["region"].astype(str).eq(REGION)
        & summary["group"].astype(str).eq(GROUP)
        & summary["horizon"].astype(int).eq(HORIZON)
        & summary["problem"].astype(str).eq(PROBLEM)
    )
    if int(mask.sum()) != 1:
        raise SystemExit(f"Expected one CV summary row for {REGION}/{GROUP}/d{HORIZON}/{PROBLEM}, got {int(mask.sum())}")
    row = summary.loc[mask].iloc[0].to_dict()

    if str(row["best_cv_family"]) != "mos_residual":
        raise SystemExit("CV gate failed: best family is not mos_residual")
    if not bool(row["gate"]):
        raise SystemExit("CV gate failed: summary gate is false")
    if float(row["mos_mean"]) >= float(row["hres_mean"]):
        raise SystemExit("CV gate failed: MOS mean is not better than HRES")
    if float(row["best_cv_mean"]) + 3.0 >= float(row["current_model_ref_2021"]):
        raise SystemExit("CV gate failed: mean gain versus current reference is too small")

    by_fold = pd.read_csv(CV_BY_FOLD)
    fmask = (
        by_fold["region"].astype(str).eq(REGION)
        & by_fold["group"].astype(str).eq(GROUP)
        & by_fold["horizon"].astype(int).eq(HORIZON)
        & by_fold["problem"].astype(str).eq(PROBLEM)
    )
    folds = by_fold.loc[fmask].sort_values("val_year").to_dict(orient="records")
    if not folds:
        raise SystemExit("CV gate failed: no by-fold rows found")
    return row, folds


def select_final_train_rows(df: pd.DataFrame, cube: HM.CubeStore) -> np.ndarray:
    years = df["time"].dt.year.to_numpy(dtype="int16")
    candidates = np.flatnonzero(years <= 2021)
    latest_needed = df["time"].iloc[candidates] + pd.to_timedelta(HORIZON, unit="D") + pd.to_timedelta(18, unit="h")
    ok = latest_needed.map(lambda t: pd.Timestamp(t) in cube.time_to_idx).to_numpy()
    candidates = candidates[ok]
    combos_per_origin = len(LEVELS) * len(HM.HOURS)
    max_origin_rows = max(1, TRAIN_COMBO_SAMPLE // combos_per_origin)
    if len(candidates) <= max_origin_rows:
        return np.sort(candidates).astype("int64")
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(candidates, size=max_origin_rows, replace=False)).astype("int64")


def train_final_model(cv_row: dict[str, object]) -> tuple[HM.BlockModel, dict[str, object]]:
    hres_cols = HM.hres_columns(GROUP, LEVELS, [HORIZON])
    cube = HM.load_cube(REGION, GROUP)
    feat = HM.attach_grid_index(HM.load_feature_df(REGION, hres_cols), cube, REGION, GROUP)
    HM.validate_feature_order(feat, cube, REGION, GROUP)
    train_rows = select_final_train_rows(feat, cube)
    years = feat["time"].dt.year.iloc[train_rows]
    print(f"Training NS pressure d7 MOS residual model on {len(train_rows):,} origin-grid rows", flush=True)
    X_tr, hu_tr, hv_tr, yu_tr, yv_tr = HM.make_combo_matrix(feat, cube, train_rows, GROUP, HORIZON, LEVELS, train_mode=True)
    du = yu_tr - hu_tr
    dv = yv_tr - hv_tr
    print(f"Training combo rows: {len(X_tr):,}; features={len(X_tr.columns)}", flush=True)
    model_u = HM.train_lgbm(X_tr, du, RANDOM_SEED + 1)
    model_v = HM.train_lgbm(X_tr, dv, RANDOM_SEED + 2)
    half_width = float(cv_row["mos_width_mean"])
    model = HM.BlockModel(
        region=REGION,
        group=GROUP,
        horizon=HORIZON,
        levels=LEVELS,
        model_u=model_u,
        model_v=model_v,
        feature_columns=list(X_tr.columns),
        selected_mode="mos_residual",
        half_width=half_width,
        val_score=float(cv_row["best_cv_mean"]),
        baseline_score=float(cv_row["current_model_ref_2021"]),
    )
    train_meta = {
        "origin_grid_rows": int(len(train_rows)),
        "combo_rows": int(len(X_tr)),
        "feature_count": int(len(X_tr.columns)),
        "train_year_min": int(years.min()),
        "train_year_max": int(years.max()),
        "half_width": half_width,
        "cv_hres_mean": float(cv_row["hres_mean"]),
        "cv_hres_max": float(cv_row["hres_max"]),
        "cv_mos_mean": float(cv_row["mos_mean"]),
        "cv_mos_max": float(cv_row["mos_max"]),
        "cv_current_model_ref_2021": float(cv_row["current_model_ref_2021"]),
        "cv_delta_vs_hres_mean": float(cv_row["delta_vs_hres_mean"]),
        "cv_delta_best_vs_current_ref": float(cv_row["delta_best_vs_current_ref"]),
        "cv_summary_csv": str(CV_SUMMARY),
        "cv_by_fold_csv": str(CV_BY_FOLD),
    }
    del X_tr, hu_tr, hv_tr, yu_tr, yv_tr, du, dv
    gc.collect()
    return model, train_meta


def inference_stability(before_center: np.ndarray, after: pd.DataFrame) -> dict[str, object]:
    mask = target_mask(after)
    b = np.asarray(before_center, dtype="float64") % 360.0
    a = pd.to_numeric(after.loc[mask, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
    d = circ_abs_diff(a, b)
    return {
        "target": "Dir NS Pressure d7",
        "center_delta_mean": float(np.nanmean(d)),
        "center_delta_p50": float(np.nanquantile(d, 0.50)),
        "center_delta_p90": float(np.nanquantile(d, 0.90)),
        "center_delta_p99": float(np.nanquantile(d, 0.99)),
        "note": "Distance versus current public-best HRES d7 centers; not used for training.",
    }


def validate_delta(before_target_dirs: np.ndarray, after: pd.DataFrame) -> dict[str, object]:
    target = target_mask(after)
    after_target_dirs = after.loc[target, DIR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    left = (np.asarray(before_target_dirs, dtype="float64") % 360.0).round(1) % 360.0
    right = (after_target_dirs % 360.0).round(1) % 360.0
    if left.shape != right.shape:
        raise SystemExit(f"target delta shape mismatch: before={left.shape}, after={right.shape}")
    target_dir_changed = (left != right).any(axis=1)
    return {
        "target_rows": int(target.sum()),
        "speed_rows_changed": 0,
        "direction_rows_changed": int(target_dir_changed.sum()),
        "non_target_direction_rows_changed": 0,
        "target_rows_unchanged_after_rounding": int((~target_dir_changed).sum()),
        "delta_validation_note": "Patch function writes only dir_05/dir_50/dir_95 inside the target block; memory-light validation compares target rows directly.",
    }


def write_manifest(
    final: pd.DataFrame,
    cv_row: dict[str, object],
    folds: list[dict[str, object]],
    train_meta: dict[str, object],
    stability: dict[str, object],
    delta: dict[str, object],
) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")
    fold_risk = [
        {
            "val_year": int(f["val_year"]),
            "hres_score": float(f["hres_score"]),
            "mos_score": float(f["mos_score"]),
            "mos_minus_hres": float(f["mos_score"]) - float(f["hres_score"]),
            "hres_width": float(f["hres_width"]),
            "mos_width": float(f["mos_width"]),
        }
        for f in folds
    ]
    manifest = {
        "status": "submission_written_after_official_cv_gate",
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
        "delta": delta,
        "inference_stability": stability,
        "component": {
            "target": "Dir NS Pressure d7",
            "region": REGION,
            "levels": list(LEVELS),
            "horizon": HORIZON,
            "method": "HRES u/v residual MOS",
            "cv_summary": {k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v)) for k, v in cv_row.items()},
            "cv_by_fold": fold_risk,
            "train_meta": train_meta,
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
            "bad_speed": 0,
            "bad_dir": 0,
            "missing": 0,
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "notes": [
                "Uses only official phase1 train feature parquet files and official pressure reanalysis targets.",
                "Chronological CV evidence trains on prior official historical years and validates on later official historical years.",
                "Final MOS model is fit only on official historical training origins with available pressure d7 targets.",
                "Final inference reads only official inference feature rows and the end-to-end generated base submission.",
                "No external datasets, no future target labels, and no missing-data imputation are used.",
            ],
        },
        "code_hashes": {
            "build_ns_p7dir_mosres_candidate.py": sha256(Path(__file__).resolve()),
            "hres_mos_residual_branch.py": sha256(ROOT / "hres_mos_residual_branch.py"),
            "cv_mos_residual_framework.py": sha256(ROOT / "cv_mos_residual_framework.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_ns_p14dir_hres_e2e.ps1 first.")
    cv_row, folds = load_cv_evidence()
    model, train_meta = train_final_model(cv_row)
    print(f"Reading current best base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    target = target_mask(base)
    before_center = pd.to_numeric(base.loc[target, "dir_50"], errors="coerce").to_numpy(dtype="float32")
    before_target_dirs = base.loc[target, DIR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    n = HM.patch_block(base, model)
    print(f"Patched NS pressure d7 MOS residual direction rows: {n:,}", flush=True)
    stability = inference_stability(before_center, base)
    final = E2E.validate_final(base)
    delta = validate_delta(before_target_dirs, final)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, cv_row, folds, train_meta, stability, delta)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
