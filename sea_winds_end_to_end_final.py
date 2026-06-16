#!/usr/bin/env python3
"""
End-to-end Sea Winds submission generator.

This is the reproducible solution entrypoint. It does not read a previous
submission CSV and patch rows. Instead it:

1. Prepares the official dataset/features.
2. Loads or trains the two official-data model profiles used by the ensemble:
   - quality_lgb_dirall
   - proper_full_refit_v1
3. Generates predictions for both profiles directly from the model bundles.
4. Assembles the validated component ensemble in memory.
5. Applies deterministic official-data postprocessors:
   - North Sea station d14 month/hour climatology.
   - North Sea surface d7 HRES direction.
   - Optional North Sea pressure d7 HRES direction.
6. Writes predictions.csv and a zip containing root-level predictions.csv.

All inputs are from `runs/v6_pressure_speed/phase1_dataset` or the official
starting-kit modules. No external weather data is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
SURFACE_LEVELS = ("10m", "100m")
HOURS = (0, 6, 12, 18)
COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
KEYS = ["type", "window", "region", "latitude", "longitude", "station", "horizon", "hour", "level"]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def load_solution_module():
    os.environ.setdefault("SEA_WINDS_WORKDIR", str(WORK))
    os.environ.setdefault("SEA_WINDS_OFFICIAL_PHASE1_DIR", str(ROOT / "external" / "Hackathon-Sea-Winds-Predictions" / "phase_1"))
    os.environ.setdefault("SEA_WINDS_KEEP_ZIP", "1")
    os.environ.setdefault("SEA_WINDS_ENABLE_STATIONS", "0")
    os.environ.setdefault("SEA_WINDS_DISABLE_STATIONS", "1")
    os.environ.setdefault("SEA_WINDS_STATION_DIR_POSTPROCESS", "1")
    os.environ.setdefault("SEA_WINDS_FINALIZE_EXISTING", "0")
    os.environ.setdefault("SEA_WINDS_N_JOBS", "4")
    path = ROOT / "sea_winds_solution_ephemeral_v6_pressure_speed.py"
    spec = importlib.util.spec_from_file_location("sea_winds_solution_v6_e2e", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SOL = load_solution_module()


def log(msg: str) -> None:
    print(msg, flush=True)


def install_vectorized_grid_predictors() -> None:
    """Replace slow row-by-row inherited inference with equivalent vectorized output."""

    def fast_predict_grid_speed_level(features_df, model_bundle_for_level, calib_for_level):
        rows = []
        lat = np.round(pd.to_numeric(features_df["latitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        lon = np.round(pd.to_numeric(features_df["longitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        n = len(features_df)
        for tgt, bundle in model_bundle_for_level.items():
            horizon = int(tgt.split("_")[1][1:])
            hour = int(tgt.split("_")[2][1:])
            feats = bundle["features"]
            X = features_df.loc[:, feats].fillna(0)

            qlo_lgb = bundle["lgb_lo"].predict(X)
            q50_lgb = bundle["lgb_mid"].predict(X)
            qhi_lgb = bundle["lgb_hi"].predict(X)
            qlo_cb = np.mean([m.predict(X) for m in bundle["cb_lo"]], axis=0) if bundle["cb_lo"] else qlo_lgb
            qhi_cb = np.mean([m.predict(X) for m in bundle["cb_hi"]], axis=0) if bundle["cb_hi"] else qhi_lgb

            cal = calib_for_level[horizon]
            w = cal["w"]
            qlo = w * qlo_cb + (1.0 - w) * qlo_lgb
            qhi = w * qhi_cb + (1.0 - w) * qhi_lgb
            q50 = q50_lgb
            qlo = q50 - cal["k_lo"] * (q50 - qlo)
            qhi = q50 + cal["k_hi"] * (qhi - q50)
            qlo = np.maximum(np.minimum(qlo, q50), 0.0)
            qhi = np.maximum(qhi, q50)

            rows.append(pd.DataFrame({
                "latitude": lat,
                "longitude": lon,
                "horizon": np.full(n, horizon, dtype=np.int16),
                "hour": np.full(n, hour, dtype=np.int8),
                "q05": qlo.astype("float32"),
                "q50": np.asarray(q50, dtype="float32"),
                "q95": qhi.astype("float32"),
            }))
        return pd.concat(rows, ignore_index=True)

    def fast_predict_grid_direction_level(features_df, model_bundle_for_level, calib_for_level):
        rows = []
        lat = np.round(pd.to_numeric(features_df["latitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        lon = np.round(pd.to_numeric(features_df["longitude"], errors="coerce").to_numpy(dtype="float64"), 2)
        n = len(features_df)
        for tgt, bundle in model_bundle_for_level.items():
            horizon = int(tgt.split("_")[1][1:])
            hour = int(tgt.split("_")[2][1:])
            feats = bundle["features"]
            X = features_df.loc[:, feats].fillna(0)
            pred_deg = (
                np.degrees(np.arctan2(bundle["sin"].predict(X), bundle["cos"].predict(X))) % 360.0
            )
            half_width = calib_for_level[horizon]["half_width"]
            rows.append(pd.DataFrame({
                "latitude": lat,
                "longitude": lon,
                "horizon": np.full(n, horizon, dtype=np.int16),
                "hour": np.full(n, hour, dtype=np.int8),
                "dir_05": ((pred_deg - half_width) % 360.0).astype("float32"),
                "dir_50": np.asarray(pred_deg, dtype="float32"),
                "dir_95": ((pred_deg + half_width) % 360.0).astype("float32"),
            }))
        return pd.concat(rows, ignore_index=True)

    SOL.predict_grid_speed_level = fast_predict_grid_speed_level
    SOL.predict_grid_direction_level = fast_predict_grid_direction_level


install_vectorized_grid_predictors()


def safe_round_dir(a: np.ndarray | pd.Series, dp: int = 3) -> np.ndarray:
    out = np.round(np.asarray(a, dtype="float64") % 360.0, dp)
    out[out >= 360.0] = 0.0
    return out


def circ_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(np.radians(arr)).mean(), np.cos(np.radians(arr)).mean())) % 360.0)


def configure_profile(profile: str, retrain_full: bool) -> None:
    cfg = SOL.CFG
    cfg.workdir = WORK
    cfg.model_profile_tag = profile
    cfg.low_memory_mode = True
    cfg.use_station_models = False
    cfg.station_dir_postprocess = True
    cfg.disable_model_cache = False
    cfg.force_retrain_models = os.environ.get("SEA_WINDS_FORCE_RETRAIN", "0") == "1"
    cfg.retrain_on_full_2019_2021 = bool(retrain_full)
    cfg.train_with_2021 = False
    cfg.random_seed = 31 if retrain_full else 42
    levels = tuple(SOL.ALL_LEVELS)
    cfg.speed_direct_levels = levels
    cfg.dir_direct_levels = levels
    cfg.direct_levels = levels
    cfg.catboost_speed_levels = tuple()
    cfg.grid_max_train_samples = int(os.environ.get("SEA_WINDS_GRID_MAX_TRAIN_SAMPLES", "320000"))
    cfg.grid_feature_subsample = int(os.environ.get("SEA_WINDS_GRID_FEATURE_SUBSAMPLE", "120000"))
    cfg.grid_dir_feature_subsample = int(os.environ.get("SEA_WINDS_GRID_DIR_FEATURE_SUBSAMPLE", "120000"))
    cfg.grid_dir_train_subsample = int(os.environ.get("SEA_WINDS_GRID_DIR_TRAIN_SUBSAMPLE", "300000"))
    cfg.lgb_speed_iterations = int(os.environ.get("SEA_WINDS_LGB_SPEED_ITERS", "1100"))
    cfg.lgb_dir_iterations = int(os.environ.get("SEA_WINDS_LGB_DIR_ITERS", "440"))
    cfg.lgb_speed_num_leaves = int(os.environ.get("SEA_WINDS_LGB_SPEED_LEAVES", "63"))
    cfg.lgb_dir_num_leaves = int(os.environ.get("SEA_WINDS_LGB_DIR_LEAVES", "63"))
    cfg.n_jobs = int(os.environ.get("SEA_WINDS_N_JOBS", "1"))


def force_single_thread_estimators(obj) -> None:
    """Avoid native LightGBM/OpenMP crashes on local Windows inference."""
    if isinstance(obj, dict):
        for v in obj.values():
            force_single_thread_estimators(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            force_single_thread_estimators(v)
        return

    get_params = getattr(obj, "get_params", None)
    set_params = getattr(obj, "set_params", None)
    if callable(get_params) and callable(set_params):
        try:
            params = get_params()
        except Exception:
            params = {}
        updates = {}
        if "n_jobs" in params:
            updates["n_jobs"] = 1
        if "thread_count" in params:
            updates["thread_count"] = 1
        if updates:
            try:
                set_params(**updates)
            except Exception:
                pass



def prepare_official_inputs():
    SOL.pip_install_if_needed()
    SOL.log_cache_configuration()
    utils_py, fe_py = SOL.download_official_modules()
    data_dir = SOL.download_and_extract_dataset()
    SOL.validate_dataset_layout(data_dir)
    utils_module = SOL.import_from_path("sea_winds_utils_official_e2e", utils_py)
    fe_module = SOL.import_from_path("sea_winds_feature_engineering_official_e2e", fe_py)
    features_dir = SOL.build_official_features(data_dir, fe_module)
    station_meta = SOL.load_station_metadata(data_dir / "scoring")
    return data_dir, features_dir, utils_module, fe_module, station_meta


def generate_profile_predictions(profile: str, retrain_full: bool, prepared) -> pd.DataFrame:
    data_dir, features_dir, utils_module, fe_module, station_meta = prepared
    configure_profile(profile, retrain_full)
    log(f"\n=== Generating profile={profile}, retrain_full={retrain_full} ===")
    frames = []
    for region in SOL.REGIONS:
        bundle = SOL.train_region_models(
            region=region,
            data_dir=data_dir,
            features_dir=features_dir,
            utils_module=utils_module,
            fe_module=fe_module,
            station_meta=station_meta,
        )
        force_single_thread_estimators(bundle)
        region_df = SOL.predict_region_all_windows(
            region=region,
            data_dir=data_dir,
            features_dir=features_dir,
            region_bundle=bundle,
            station_meta=station_meta,
            utils_module=utils_module,
            fe_module=fe_module,
        )
        frames.append(region_df)
    raw = pd.concat(frames, ignore_index=True)
    return SOL.format_full_codabench_submission(raw)


def normalize_for_assembly(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    return out.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def assert_same_keys(a: pd.DataFrame, b: pd.DataFrame) -> None:
    lhs = a[KEYS].fillna("__NA__").astype(str)
    rhs = b[KEYS].fillna("__NA__").astype(str)
    if not bool((lhs.values == rhs.values).all()):
        raise RuntimeError("Generated component profiles do not have identical submission keys")


def assemble_component_ensemble(base: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Assemble validated model-profile components in memory."""
    df = normalize_for_assembly(base)
    full = normalize_for_assembly(full)
    assert_same_keys(df, full)

    grid = df["type"].eq("grid")
    station = df["type"].eq("station")

    def take(mask: pd.Series, cols: Iterable[str], label: str) -> None:
        idx = df.index[mask]
        df.loc[idx, list(cols)] = full.loc[idx, list(cols)].to_numpy()
        log(f"  component {label}: {len(idx):,} rows")

    log("\nAssembling validated component ensemble")
    take(grid & df["region"].eq("north_sea") & df["level"].isin(SURFACE_LEVELS) & df["horizon"].isin([1, 7, 14]), SPEED_COLS, "speed_ns_surface_d1_d7_d14")
    take(grid & df["region"].eq("north_sea") & df["level"].isin(PRESSURE_LEVELS) & df["horizon"].isin([1, 7, 14]), SPEED_COLS, "speed_ns_pressure_d1_d7_d14")
    take(grid & df["region"].eq("east_china_sea") & df["level"].isin(SURFACE_LEVELS) & df["horizon"].isin([1, 14]), SPEED_COLS, "speed_ecs_surface_d1_d14")
    take(grid & df["region"].eq("east_china_sea") & df["level"].isin(PRESSURE_LEVELS) & df["horizon"].eq(7), SPEED_COLS, "speed_ecs_pressure_d7")

    # Direction blocks validated from the full-refit profile. North Sea station
    # d14 and surface d7 are overwritten later by stronger deterministic
    # official-data components.
    take(grid & df["region"].eq("north_sea") & df["level"].isin(SURFACE_LEVELS) & df["horizon"].eq(14), DIR_COLS, "dir_ns_surface_d14")
    take(grid & df["region"].eq("north_sea") & df["level"].isin(PRESSURE_LEVELS) & df["horizon"].eq(1), DIR_COLS, "dir_ns_pressure_d1")
    take(grid & df["region"].eq("east_china_sea") & df["level"].isin(SURFACE_LEVELS) & df["horizon"].eq(1), DIR_COLS, "dir_ecs_surface_d1")
    take(grid & df["region"].eq("east_china_sea") & df["level"].isin(PRESSURE_LEVELS) & df["horizon"].eq(1), DIR_COLS, "dir_ecs_pressure_d1")
    return df


def read_anchor(window: int) -> pd.Timestamp:
    meta = json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text())
    return pd.Timestamp(meta["context_end"])


def load_station_history(region: str, window: int) -> pd.DataFrame:
    train = pd.read_parquet(DATA / "train" / f"stations_{region}_6h.parquet")
    ctx = pd.read_parquet(DATA / "inference" / f"window_{window}" / f"context_stations_{region}.parquet")
    hist = pd.concat([train, ctx], ignore_index=True)
    hist["time"] = pd.to_datetime(hist["time"])
    hist["station"] = hist["station"].astype(str)
    hist["direction"] = pd.to_numeric(hist["direction"], errors="coerce") % 360.0
    hist["hour"] = hist["time"].dt.hour.astype("int8")
    hist["month"] = hist["time"].dt.month.astype("int8")
    return hist


def month_clim_direction(hist: pd.DataFrame, station: str, anchor: pd.Timestamp, hour: int, target_time: pd.Timestamp) -> float:
    sub = hist[
        hist["station"].eq(station)
        & hist["hour"].eq(hour)
        & hist["time"].dt.year.lt(anchor.year)
        & hist["month"].eq(target_time.month)
    ]
    return circ_mean(sub["direction"])


def apply_ns_station_d14_month_clim(df: pd.DataFrame) -> int:
    count = 0
    for window in range(1, 9):
        anchor = read_anchor(window)
        hist = load_station_history("north_sea", window)
        mask = (
            df["type"].eq("station")
            & df["window"].eq(window)
            & df["region"].eq("north_sea")
            & df["horizon"].eq(14)
        )
        for idx in df.index[mask]:
            station = str(df.at[idx, "station"])
            hour = int(df.at[idx, "hour"])
            target_time = anchor + pd.Timedelta(days=14) + pd.Timedelta(hours=hour)
            center = month_clim_direction(hist, station, anchor, hour, target_time)
            if not np.isfinite(center):
                continue
            hw = 135.0
            df.at[idx, "dir_50"] = center % 360.0
            df.at[idx, "dir_05"] = (center - hw) % 360.0
            df.at[idx, "dir_95"] = (center + hw) % 360.0
            count += 1
    log(f"  deterministic ns_station_d14_month_clim: {count:,} rows")
    return count


def apply_ns_surface_d7_hres(df: pd.DataFrame) -> int:
    count = 0
    for window in range(1, 9):
        inf = pd.read_parquet(
            FEATURES / f"inference_window_{window}_north_sea.parquet",
            columns=["latitude", "longitude"] + [f"fcst_dir_d7_h{h}" for h in HOURS],
        )
        inf["latitude"] = inf["latitude"].astype("float32").round(2)
        inf["longitude"] = inf["longitude"].astype("float32").round(2)
        for hour in HOURS:
            centers = inf[["latitude", "longitude"]].copy()
            centers["center"] = pd.to_numeric(inf[f"fcst_dir_d7_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0
            for level in SURFACE_LEVELS:
                mask = (
                    df["type"].eq("grid")
                    & df["window"].eq(window)
                    & df["region"].eq("north_sea")
                    & df["horizon"].eq(7)
                    & df["hour"].eq(hour)
                    & df["level"].eq(level)
                )
                idx = df.index[mask]
                merged = df.loc[idx, ["latitude", "longitude"]].merge(centers, on=["latitude", "longitude"], how="left", sort=False)
                if len(merged) != len(idx) or merged["center"].isna().any():
                    raise RuntimeError(f"missing NS surface d7 HRES centers for W{window} h{hour} {level}")
                center = merged["center"].to_numpy(dtype="float64")
                hw = 135.0
                df.loc[idx, "dir_50"] = center
                df.loc[idx, "dir_05"] = (center - hw) % 360.0
                df.loc[idx, "dir_95"] = (center + hw) % 360.0
                count += len(idx)
    log(f"  deterministic ns_surface_d7_hres: {count:,} rows")
    return count


def apply_ns_pressure_d7_hres(df: pd.DataFrame) -> int:
    count = 0
    hres_cols = ["latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        for hour in HOURS:
            hres_cols += [f"fcst_u_{level}_d7_h{hour}", f"fcst_v_{level}_d7_h{hour}"]
    for window in range(1, 9):
        inf = pd.read_parquet(FEATURES / f"inference_window_{window}_north_sea.parquet", columns=hres_cols)
        inf["latitude"] = inf["latitude"].astype("float32").round(2)
        inf["longitude"] = inf["longitude"].astype("float32").round(2)
        for level in PRESSURE_LEVELS:
            for hour in HOURS:
                u = pd.to_numeric(inf[f"fcst_u_{level}_d7_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                v = pd.to_numeric(inf[f"fcst_v_{level}_d7_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                centers = inf[["latitude", "longitude"]].copy()
                centers["center"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
                mask = (
                    df["type"].eq("grid")
                    & df["window"].eq(window)
                    & df["region"].eq("north_sea")
                    & df["horizon"].eq(7)
                    & df["hour"].eq(hour)
                    & df["level"].eq(level)
                )
                idx = df.index[mask]
                merged = df.loc[idx, ["latitude", "longitude"]].merge(centers, on=["latitude", "longitude"], how="left", sort=False)
                if len(merged) != len(idx) or merged["center"].isna().any():
                    raise RuntimeError(f"missing NS pressure d7 HRES centers for W{window} h{hour} {level}")
                center = merged["center"].to_numpy(dtype="float64")
                hw = 140.0
                df.loc[idx, "dir_50"] = center
                df.loc[idx, "dir_05"] = (center - hw) % 360.0
                df.loc[idx, "dir_95"] = (center + hw) % 360.0
                count += len(idx)
    log(f"  deterministic ns_pressure_d7_hres: {count:,} rows")
    return count


def validate_final(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    for c in SPEED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0).round(2)
    out["q05"] = out[["q05", "q50"]].min(axis=1).round(2)
    out["q95"] = out[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        out[c] = safe_round_dir(out[c], dp=3)

    grid = out["type"].eq("grid")
    station = out["type"].eq("station")
    bad_speed = ((out["q05"] > out["q50"]) | (out["q50"] > out["q95"]) | (out[SPEED_COLS] < 0).any(axis=1)).sum()
    bad_dir = ((out[DIR_COLS] < 0) | (out[DIR_COLS] >= 360) | out[DIR_COLS].isna()).any(axis=1).sum()
    grid_dup = out.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum()
    station_dup = out.loc[station].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum()
    counts = out["type"].value_counts(dropna=False).to_dict()
    log(
        f"Validation rows={len(out):,} counts={counts} bad_speed={int(bad_speed)} "
        f"bad_dir={int(bad_dir)} grid_dup={int(grid_dup)} station_dup={int(station_dup)}"
    )
    if len(out) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise RuntimeError("row/type count validation failed")
    if bad_speed or bad_dir or grid_dup or station_dup or out[SPEED_COLS + DIR_COLS].isna().any().any():
        raise RuntimeError("content validation failed")
    return out.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _trim_decimal(value, decimals: int) -> str:
    if pd.isna(value):
        return ""
    text = f"{float(value):.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def compact_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    for col in ["latitude", "longitude"]:
        vals = pd.to_numeric(out[col], errors="coerce")
        out[col] = [_trim_decimal(v, 2) for v in vals]
    for col in SPEED_COLS:
        vals = pd.to_numeric(out[col], errors="coerce")
        out[col] = [_trim_decimal(v, 2) for v in vals]
    for col in DIR_COLS:
        vals = pd.to_numeric(out[col], errors="coerce") % 360.0
        out[col] = [_trim_decimal(v, 1) for v in vals]
    for col in ["window", "horizon", "hour"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("int64").astype(str)
    for col in ["type", "region", "station", "level"]:
        out[col] = out[col].fillna("").astype(str).replace({"nan": "", "None": ""})
    return out


def _write_compact_csv_chunk(frame: pd.DataFrame, handle, header: bool) -> None:
    kwargs = {"index": False, "header": header, "lineterminator": "\r\n"}
    try:
        frame.to_csv(handle, **kwargs)
    except TypeError:
        kwargs.pop("lineterminator")
        frame.to_csv(handle, line_terminator="\r\n", **kwargs)


def write_zip(df: pd.DataFrame, output_csv: Path, output_zip: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    chunk_rows = int(os.environ.get("SEA_WINDS_WRITE_CHUNK_ROWS", "200000"))
    tmp_csv = output_csv.with_name(output_csv.name + ".tmp")
    tmp_zip = output_zip.with_name(output_zip.name + ".tmp")
    for tmp_path in (tmp_csv, tmp_zip):
        if tmp_path.exists():
            tmp_path.unlink()

    wrote_header = False
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        for start in range(0, len(df), chunk_rows):
            compact = compact_csv_frame(df.iloc[start : start + chunk_rows])
            _write_compact_csv_chunk(compact, handle, header=not wrote_header)
            wrote_header = True

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as zf:
        zf.write(tmp_csv, arcname="predictions.csv")

    with zipfile.ZipFile(tmp_zip) as zf:
        info = zf.getinfo("predictions.csv")
        names = zf.namelist()
    if names != ["predictions.csv"] or info.file_size != tmp_csv.stat().st_size:
        raise RuntimeError("zip validation failed")

    os.replace(tmp_csv, output_csv)
    os.replace(tmp_zip, output_zip)
    log(f"Wrote {output_csv} ({output_csv.stat().st_size:,} bytes)")
    log(f"Wrote {output_zip} ({output_zip.stat().st_size:,} bytes), names={names}, uncompressed={info.file_size:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ns-pressure-d7", action="store_true", help="also include the validated NS pressure d7 HRES direction component")
    parser.add_argument("--output-csv", type=Path, default=WORK / "predictions_end_to_end_final_compact.csv")
    parser.add_argument("--output-zip", type=Path, default=WORK / "submission_end_to_end_final_compact.zip")
    args = parser.parse_args()

    prepared = prepare_official_inputs()
    base = generate_profile_predictions("quality_lgb_dirall", retrain_full=False, prepared=prepared)
    full = generate_profile_predictions("proper_full_refit_v1", retrain_full=True, prepared=prepared)
    final = assemble_component_ensemble(base, full)
    log("\nApplying deterministic official-data components")
    apply_ns_station_d14_month_clim(final)
    apply_ns_surface_d7_hres(final)
    if args.include_ns_pressure_d7:
        apply_ns_pressure_d7_hres(final)
    final = validate_final(final)
    write_zip(final, args.output_csv, args.output_zip)


if __name__ == "__main__":
    main()
