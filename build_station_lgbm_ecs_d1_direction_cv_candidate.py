from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import station_cv_mos_analog_framework as CV


WORK = Path("runs/v6_pressure_speed")
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"
BASE_CSV = WORK / "predictions_public_positive_fullrefit_hybrid_compact.csv"
OUT_CSV = WORK / "predictions_station_lgbm_ecs_d1_dir_cv_compact.csv"
OUT_ZIP = WORK / "submission_station_lgbm_ecs_d1_dir_cv_compact.zip"
SUMMARY_CSV = WORK / "station_lgbm_ecs_d1_dir_cv_summary.csv"

REGION = "east_china_sea"
HORIZON = 1
LGB_ESTIMATORS = 140
SEED = 20260525

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


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


def fit_direction_models(train: pd.DataFrame, feats: list[str], seed: int) -> tuple[object, object]:
    ok = np.isfinite(train["y_dir"].to_numpy(dtype="float64")) & np.isfinite(train["hres_dir"].to_numpy(dtype="float64"))
    X = train.loc[ok, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    h = np.radians(train.loc[ok, "hres_dir"].to_numpy(dtype="float64") % 360.0)
    y = np.radians(train.loc[ok, "y_dir"].to_numpy(dtype="float64") % 360.0)
    mx = CV.fit_lgbm(X, np.cos(y) - np.cos(h), "regression", seed + 100, LGB_ESTIMATORS)
    my = CV.fit_lgbm(X, np.sin(y) - np.sin(h), "regression", seed + 101, LGB_ESTIMATORS)
    return mx, my


def predict_direction(models: tuple[object, object], feats: list[str], df: pd.DataFrame) -> np.ndarray:
    mx, my = models
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    h = np.radians(df["hres_dir"].to_numpy(dtype="float64") % 360.0)
    px = np.cos(h) + mx.predict(X).astype("float64")
    py = np.sin(h) + my.predict(X).astype("float64")
    return np.degrees(np.arctan2(py, px)) % 360.0


def calibrate_width(train_base: pd.DataFrame, hist: CV.StationHistory) -> tuple[float, pd.DataFrame]:
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    rows = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year) & df["y_dir"].notna()].copy()
        val = df[CV.anchor_mask(df, val_year) & df["y_dir"].notna()].copy()
        feats = CV.numeric_features(train)
        models = fit_direction_models(train, feats, SEED + val_year)
        pred = predict_direction(models, feats, val)
        y = val["y_dir"].to_numpy(dtype="float64") % 360.0
        score, width = CV.best_direction_interval(y, pred)
        rows.append({"val_year": val_year, "score": score, "half_width": width, "train_rows": len(train), "val_rows": len(val)})
        print(f"CV {REGION} station d{HORIZON} dir {val_year}: score={score:.4f} half_width={width:.1f}", flush=True)
    summary = pd.DataFrame(rows)
    return float(summary["half_width"].mean()), summary


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
    half_width, cv_summary = calibrate_width(train_base, train_hist)
    cv_summary["selected_half_width"] = half_width
    cv_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Selected half_width={half_width:.1f}; wrote {SUMMARY_CSV}", flush=True)

    train_df = CV.build_combo(train_base, train_hist, HORIZON, include_climatology=False)
    train_df = train_df[train_df["y_dir"].notna()].copy()
    feats = CV.numeric_features(train_df)
    models = fit_direction_models(train_df, feats, SEED + 999)

    patch_rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(REGION, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(REGION, window))
        for hour in CV.HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            pred = predict_direction(models, feats, inf)
            for station, center in zip(inf["station"].astype(str), pred):
                patch_rows.append(
                    {
                        "window": window,
                        "region": REGION,
                        "station": station,
                        "horizon": HORIZON,
                        "hour": int(hour),
                        "dir_05_new": float((center - half_width) % 360.0),
                        "dir_50_new": float(center % 360.0),
                        "dir_95_new": float((center + half_width) % 360.0),
                    }
                )
    patch = pd.DataFrame(patch_rows)
    key = ["window", "region", "station", "horizon", "hour"]
    merged = out.reset_index().merge(patch, on=key, how="left", validate="many_to_one")
    station_mask = (
        merged["type"].eq("station")
        & merged["region"].eq(REGION)
        & merged["horizon"].eq(HORIZON)
        & merged["dir_50_new"].notna()
    )
    changed = int(station_mask.sum())
    if changed != 224:
        raise SystemExit(f"expected 224 ECS station d1 direction rows, got {changed}")
    before_speed = out[SPEED_COLS].round(2).copy()
    for c in DIR_COLS:
        merged.loc[station_mask, c] = merged.loc[station_mask, f"{c}_new"]
    out2 = merged.drop(columns=["dir_05_new", "dir_50_new", "dir_95_new"]).set_index("index").sort_index()[COLS]
    speed_changed = int((before_speed.to_numpy() != out2[SPEED_COLS].round(2).to_numpy()).any(axis=1).sum())
    dir_changed = int((out[DIR_COLS].round(1).to_numpy() != out2[DIR_COLS].round(1).to_numpy()).any(axis=1).sum())
    print(f"Patched rows={changed}; speed_changed={speed_changed}; direction_changed={dir_changed}", flush=True)
    if speed_changed != 0 or dir_changed != 224:
        raise SystemExit("unexpected delta outside ECS station d1 direction")

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
    print(f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; uncompressed={info.file_size:,}; names={names}; testzip={bad}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
