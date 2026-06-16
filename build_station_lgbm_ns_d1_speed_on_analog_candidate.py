from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import station_cv_mos_analog_framework as CV


WORK = Path("runs/v6_pressure_speed")
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"
BASE_CSV = WORK / "predictions_ns_surface_d14_speed_analog_mos_v1_compact.csv"
OUT_CSV = WORK / "predictions_station_lgbm_ns_d1_speed_on_analog_compact.csv"
OUT_ZIP = WORK / "submission_station_lgbm_ns_d1_speed_on_analog_compact.zip"
SUMMARY_CSV = WORK / "station_lgbm_ns_d1_speed_on_analog_cv_summary.csv"
MANIFEST = WORK / "station_lgbm_ns_d1_speed_on_analog_manifest.json"

REGION = "north_sea"
HORIZON = 1
LGB_ESTIMATORS = 320
SEED = 20260527
EXPECTED_PATCH_ROWS = 256

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inference_origin_rows(region: str, meta: pd.DataFrame, window: int) -> pd.DataFrame:
    meta_r = meta[meta["region"].eq(region)].copy()
    coords = meta_r[["nearest_grid_lat", "nearest_grid_lon"]].drop_duplicates().rename(
        columns={"nearest_grid_lat": "latitude", "nearest_grid_lon": "longitude"}
    )
    path = FEATURES / f"inference_window_{window}_{region}.parquet"
    available = set(CV.schema_names(path))
    cols = ["time", "latitude", "longitude"]
    cols += [c for c in CV.BASE_FEATURES if c in available]
    cols += [c for c in CV.hres_columns() if c in available]
    cols = list(dict.fromkeys(cols))
    feat = pd.read_parquet(path, columns=cols)
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = pd.to_numeric(feat["latitude"], errors="coerce").round(2)
    feat["longitude"] = pd.to_numeric(feat["longitude"], errors="coerce").round(2)
    feat = feat.merge(coords, on=["latitude", "longitude"], how="inner")
    rows = meta_r.merge(
        feat,
        left_on=["nearest_grid_lat", "nearest_grid_lon"],
        right_on=["latitude", "longitude"],
        how="inner",
        suffixes=("_station", "_grid"),
    )
    rows = rows.rename(columns={"latitude_station": "station_lat", "longitude_station": "station_lon"})
    rows["station_code"] = rows["station"].astype("category").cat.codes.astype("int16")
    rows["origin_year"] = rows["time"].dt.year.astype("int16")
    rows["origin_month"] = rows["time"].dt.month.astype("int8")
    rows["origin_doy"] = rows["time"].dt.dayofyear.astype("int16")
    rows["origin_doy_sin"] = np.sin(2.0 * np.pi * rows["origin_doy"].astype(float) / 366.0)
    rows["origin_doy_cos"] = np.cos(2.0 * np.pi * rows["origin_doy"].astype(float) / 366.0)
    rows["window"] = int(window)
    return rows.reset_index(drop=True)


def load_station_obs_with_context(region: str, window: int) -> pd.DataFrame:
    train = pd.read_parquet(DATA / "train" / f"stations_{region}_6h.parquet")
    ctx = pd.read_parquet(DATA / "inference" / f"window_{window}" / f"context_stations_{region}.parquet")
    df = pd.concat([train, ctx], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df["station"] = df["station"].astype(str)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360.0
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    return df.sort_values(["station", "time"]).drop_duplicates(["station", "time"], keep="last").reset_index(drop=True)


def fit_speed_models(train: pd.DataFrame, feats: list[str], seed: int) -> tuple[object, object, object]:
    ok = np.isfinite(train["y_speed"].to_numpy(dtype="float64"))
    X = train.loc[ok, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    y = train.loc[ok, "y_speed"].to_numpy(dtype="float64")
    m05 = CV.fit_lgbm(X, y, "quantile", seed + 1, LGB_ESTIMATORS, 0.05)
    m50 = CV.fit_lgbm(X, y, "quantile", seed + 2, LGB_ESTIMATORS, 0.50)
    m95 = CV.fit_lgbm(X, y, "quantile", seed + 3, LGB_ESTIMATORS, 0.95)
    return m05, m50, m95


def predict_speed(models: tuple[object, object, object], feats: list[str], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m05, m50, m95 = models
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    q05 = m05.predict(X).astype("float64")
    q50 = m50.predict(X).astype("float64")
    q95 = m95.predict(X).astype("float64")
    lo = np.maximum(0.0, np.minimum(q05, q50))
    mid = np.maximum(0.0, q50)
    hi = np.maximum(q95, mid)
    return lo, mid, hi


def optimize_speed_calibration(y: np.ndarray, lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    best = {"score": float("inf"), "bias": 0.0, "k_lo": 1.0, "k_hi": 1.0}
    lower = np.maximum(mid - lo, 0.05)
    upper = np.maximum(hi - mid, 0.05)
    for bias in (-0.75, -0.50, -0.30, -0.15, 0.0, 0.15, 0.30, 0.50, 0.75):
        center = np.maximum(0.0, mid + bias)
        for k_lo in (0.55, 0.70, 0.85, 1.0, 1.20, 1.45, 1.75, 2.10, 2.60):
            for k_hi in (0.55, 0.70, 0.85, 1.0, 1.20, 1.45, 1.75, 2.10, 2.60):
                plo = np.maximum(0.0, center - k_lo * lower)
                phi = np.maximum(center, center + k_hi * upper)
                score = CV.speed_winkler(y, plo, phi)
                if score < best["score"]:
                    best = {
                        "score": float(score),
                        "bias": float(bias),
                        "k_lo": float(k_lo),
                        "k_hi": float(k_hi),
                    }
    return best


def apply_calibration(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, cal: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.maximum(mid - lo, 0.05)
    upper = np.maximum(hi - mid, 0.05)
    center = np.maximum(0.0, mid + float(cal["bias"]))
    plo = np.maximum(0.0, center - float(cal["k_lo"]) * lower)
    phi = np.maximum(center, center + float(cal["k_hi"]) * upper)
    return plo, center, phi


def calibrate_speed(train_base: pd.DataFrame, hist: CV.StationHistory) -> tuple[dict[str, float], pd.DataFrame]:
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    fold_rows = []
    pooled = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year) & df["y_speed"].notna()].copy()
        val = df[CV.anchor_mask(df, val_year) & df["y_speed"].notna()].copy()
        feats = CV.numeric_features(train)
        models = fit_speed_models(train, feats, SEED + val_year)
        lo, mid, hi = predict_speed(models, feats, val)
        y = val["y_speed"].to_numpy(dtype="float64")
        raw = CV.speed_winkler(y, lo, hi)
        cal = optimize_speed_calibration(y, lo, mid, hi)
        clo, cmid, chi = apply_calibration(lo, mid, hi, cal)
        tuned = CV.speed_winkler(y, clo, chi)
        fold_rows.append(
            {
                "val_year": val_year,
                "raw_score": raw,
                "tuned_score": tuned,
                "raw_width": float(np.nanmean(hi - lo)),
                "tuned_width": float(np.nanmean(chi - clo)),
                "bias": cal["bias"],
                "k_lo": cal["k_lo"],
                "k_hi": cal["k_hi"],
                "train_rows": len(train),
                "val_rows": len(val),
            }
        )
        pooled.append(pd.DataFrame({"y": y, "lo": lo, "mid": mid, "hi": hi}))
        print(
            f"CV {REGION} station d{HORIZON} speed {val_year}: "
            f"raw={raw:.4f} tuned={tuned:.4f} width={float(np.nanmean(chi - clo)):.3f}",
            flush=True,
        )
    pool = pd.concat(pooled, ignore_index=True)
    pooled_cal = optimize_speed_calibration(
        pool["y"].to_numpy(dtype="float64"),
        pool["lo"].to_numpy(dtype="float64"),
        pool["mid"].to_numpy(dtype="float64"),
        pool["hi"].to_numpy(dtype="float64"),
    )
    summary = pd.DataFrame(fold_rows)
    summary["selected_bias"] = pooled_cal["bias"]
    summary["selected_k_lo"] = pooled_cal["k_lo"]
    summary["selected_k_hi"] = pooled_cal["k_hi"]
    summary["selected_pooled_cv_score"] = pooled_cal["score"]
    return pooled_cal, summary


def validate_submission(df: pd.DataFrame) -> None:
    for c in SPEED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).round(2)
    df["q05"] = df[["q05", "q50"]].min(axis=1).round(2)
    df["q95"] = df[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)
    grid = df["type"].eq("grid")
    counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    grid_dup = int(df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(df.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    print(f"rows={len(df):,}; counts={counts}", flush=True)
    print(f"bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    if len(df) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise SystemExit("row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise SystemExit("content validation failed")


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
    calibration, cv_summary = calibrate_speed(train_base, train_hist)
    cv_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Selected calibration={calibration}; wrote {SUMMARY_CSV}", flush=True)

    train_df = CV.build_combo(train_base, train_hist, HORIZON, include_climatology=False)
    train_df = train_df[train_df["y_speed"].notna()].copy()
    feats = CV.numeric_features(train_df)
    models = fit_speed_models(train_df, feats, SEED + 999)

    patch_rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            lo, mid, hi = predict_speed(models, feats, inf)
            lo, mid, hi = apply_calibration(lo, mid, hi, calibration)
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
    patch = pd.DataFrame(patch_rows)
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
            "selected_calibration": calibration,
            "val_rows": cv_summary.to_dict(orient="records"),
            "independent_cv_reference": "runs/v6_pressure_speed/station_cv_mos_analog/station_cv_mos_analog_ns_d1_speed_lgbm_only_summary.csv",
            "independent_cv_lgbm_score_mean": 7.667499,
            "independent_cv_lgbm_score_max": 8.051246,
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
            "The base submission is regenerated by the existing analog MOS end-to-end branch.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
