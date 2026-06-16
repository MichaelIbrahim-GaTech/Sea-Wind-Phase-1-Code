from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path("runs/v6_pressure_speed")
FEATURES = WORK / "phase1_dataset" / "features"
BASE_CSV = WORK / "predictions_pressurefix_station_long_calibrated_compact.csv"
MODEL_CSV = WORK / "predictions_direction_all_station_refine_compact.csv"
OUT_CSV = WORK / "predictions_broad_grid_dir_anchorblend_keepwidth_compact.csv"
OUT_ZIP = WORK / "submission_broad_grid_dir_anchorblend_keepwidth_compact.zip"

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]
GRID_KEY = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]
PRESSURE_LEVELS = ["1000", "925", "850", "700", "500"]
HOURS = [0, 6, 12, 18]


def normalize(df: pd.DataFrame) -> None:
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)


def blend_deg(a, b, w_b: float) -> np.ndarray:
    a = np.asarray(a, dtype="float64") % 360.0
    b = np.asarray(b, dtype="float64") % 360.0
    ar = np.radians(a)
    br = np.radians(b)
    x = (1.0 - w_b) * np.cos(ar) + w_b * np.cos(br)
    y = (1.0 - w_b) * np.sin(ar) + w_b * np.sin(br)
    return np.degrees(np.arctan2(y, x)) % 360.0


def recenter(df: pd.DataFrame, mask: pd.Series, center) -> int:
    idx = df.index[mask]
    old_lo = pd.to_numeric(df.loc[idx, "dir_05"], errors="coerce").to_numpy(dtype="float64") % 360.0
    old_hi = pd.to_numeric(df.loc[idx, "dir_95"], errors="coerce").to_numpy(dtype="float64") % 360.0
    half_width = ((old_hi - old_lo) % 360.0) / 2.0
    center = np.asarray(center, dtype="float64") % 360.0
    df.loc[idx, "dir_50"] = center
    df.loc[idx, "dir_05"] = (center - half_width) % 360.0
    df.loc[idx, "dir_95"] = (center + half_width) % 360.0
    return len(idx)


def load_hres_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    surface_rows = []
    pressure_rows = []
    for window in range(1, 9):
        for region in ["north_sea", "east_china_sea"]:
            cols = ["latitude", "longitude"]
            for hr in HOURS:
                cols.append(f"fcst_dir_d10_h{hr}")
                for lev in PRESSURE_LEVELS:
                    cols.extend([f"fcst_u_{lev}_d7_h{hr}", f"fcst_v_{lev}_d7_h{hr}"])
            feat = pd.read_parquet(FEATURES / f"inference_window_{window}_{region}.parquet", columns=cols)
            feat["window"] = window
            feat["region"] = region
            feat["latitude"] = feat["latitude"].astype(float).round(2)
            feat["longitude"] = feat["longitude"].astype(float).round(2)

            for hr in HOURS:
                surf = feat[["window", "region", "latitude", "longitude", f"fcst_dir_d10_h{hr}"]].copy()
                surf["horizon"] = 14
                surf["hour"] = hr
                surf["hres_surface_d14"] = pd.to_numeric(surf[f"fcst_dir_d10_h{hr}"], errors="coerce") % 360.0
                surface_rows.append(surf[["window", "region", "latitude", "longitude", "horizon", "hour", "hres_surface_d14"]])

            for lev in PRESSURE_LEVELS:
                for hr in HOURS:
                    u = pd.to_numeric(feat[f"fcst_u_{lev}_d7_h{hr}"], errors="coerce").to_numpy(dtype="float64")
                    v = pd.to_numeric(feat[f"fcst_v_{lev}_d7_h{hr}"], errors="coerce").to_numpy(dtype="float64")
                    pres = feat[["window", "region", "latitude", "longitude"]].copy()
                    pres["horizon"] = 7
                    pres["hour"] = hr
                    pres["level"] = lev
                    pres["hres_pressure_d7"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
                    pressure_rows.append(pres)

    surface = pd.concat(surface_rows, ignore_index=True).drop_duplicates(
        ["window", "region", "latitude", "longitude", "horizon", "hour"]
    )
    pressure = pd.concat(pressure_rows, ignore_index=True).drop_duplicates(
        ["window", "region", "latitude", "longitude", "horizon", "hour", "level"]
    )
    return surface, pressure


def main() -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = pd.read_csv(BASE_CSV, low_memory=False)[COLS].copy()
    print(f"Reading model centers {MODEL_CSV} ({MODEL_CSV.stat().st_size:,} bytes)", flush=True)
    model = pd.read_csv(MODEL_CSV, usecols=GRID_KEY + ["dir_50"], low_memory=False)
    normalize(base)
    normalize(model)
    base["station"] = base["station"].fillna("").astype(str)
    model["dir_50"] = pd.to_numeric(model["dir_50"], errors="coerce") % 360.0

    grid = base["type"].eq("grid")
    model_grid = model[model["type"].eq("grid")].copy()
    base["model_center"] = np.nan
    grid_lookup = base.loc[grid].reset_index()[["index"] + GRID_KEY]
    merged = grid_lookup.merge(model_grid, on=GRID_KEY, how="left", validate="one_to_one")
    if merged["dir_50"].isna().any():
        raise SystemExit(f"missing grid model centers: {int(merged['dir_50'].isna().sum())}")
    base.loc[merged["index"].to_numpy(), "model_center"] = merged["dir_50"].to_numpy()

    base["model700_center"] = np.nan
    proxy_key = ["type", "window", "region", "latitude", "longitude", "horizon", "hour"]
    model700 = model_grid[model_grid["level"].eq("700")][proxy_key + ["dir_50"]].rename(columns={"dir_50": "model700_center"})
    merged700 = grid_lookup.merge(model700, on=proxy_key, how="left", validate="many_to_one")
    base.loc[merged700["index"].to_numpy(), "model700_center"] = merged700["model700_center"].to_numpy()

    print("Loading HRES inference direction maps", flush=True)
    hres_surface, hres_pressure = load_hres_maps()
    base["hres_surface_d14"] = np.nan
    merged_surf = base.loc[grid].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour"]].merge(
        hres_surface,
        on=["window", "region", "latitude", "longitude", "horizon", "hour"],
        how="left",
        validate="many_to_one",
    )
    base.loc[merged_surf["index"].to_numpy(), "hres_surface_d14"] = merged_surf["hres_surface_d14"].to_numpy()

    base["hres_pressure_d7"] = np.nan
    merged_pres = base.loc[grid].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]].merge(
        hres_pressure,
        on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
        how="left",
        validate="many_to_one",
    )
    base.loc[merged_pres["index"].to_numpy(), "hres_pressure_d7"] = merged_pres["hres_pressure_d7"].to_numpy()

    patch_counts = {}
    m = grid & base["region"].eq("north_sea") & base["level"].isin(["10m", "100m"]) & base["horizon"].eq(14)
    patch_counts["ns_surface_d14_model700blend025"] = recenter(
        base, m, blend_deg(base.loc[m, "model_center"], base.loc[m, "model700_center"], 0.25)
    )

    m = grid & base["region"].eq("east_china_sea") & base["level"].isin(["10m", "100m"]) & base["horizon"].eq(14)
    if base.loc[m, "hres_surface_d14"].isna().any():
        raise SystemExit("missing ECS surface HRES d14")
    patch_counts["ecs_surface_d14_hresblend025"] = recenter(
        base, m, blend_deg(base.loc[m, "model_center"], base.loc[m, "hres_surface_d14"], 0.25)
    )

    m = grid & base["region"].eq("north_sea") & base["level"].isin(PRESSURE_LEVELS) & base["horizon"].eq(7)
    if base.loc[m, "hres_pressure_d7"].isna().any():
        raise SystemExit("missing NS pressure HRES d7")
    patch_counts["ns_pressure_d7_hres_center"] = recenter(base, m, base.loc[m, "hres_pressure_d7"])

    m = grid & base["region"].eq("east_china_sea") & base["level"].isin(PRESSURE_LEVELS) & base["horizon"].eq(7)
    if base.loc[m, "hres_pressure_d7"].isna().any():
        raise SystemExit("missing ECS pressure HRES d7")
    patch_counts["ecs_pressure_d7_hresblend075"] = recenter(
        base, m, blend_deg(base.loc[m, "model_center"], base.loc[m, "hres_pressure_d7"], 0.75)
    )

    base = base.drop(columns=["model_center", "model700_center", "hres_surface_d14", "hres_pressure_d7"])
    for c in ["q05", "q50", "q95"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").clip(lower=0).round(2)
    base["q05"] = base[["q05", "q50"]].min(axis=1).round(2)
    base["q95"] = base[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        base[c] = ((pd.to_numeric(base[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)

    print(f"Patch counts: {patch_counts}", flush=True)
    print(f"Writing {OUT_CSV}", flush=True)
    base.to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")

    counts = base["type"].value_counts(dropna=False).to_dict()
    grid = base["type"].eq("grid")
    bad_speed = int(((base["q05"] > base["q50"]) | (base["q50"] > base["q95"]) | (base[["q05", "q50", "q95"]] < 0).any(axis=1)).sum())
    bad_dir = int(((base[DIR_COLS] < 0) | (base[DIR_COLS] >= 360) | base[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(base[["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]].isna().any(axis=1).sum())
    grid_dup = int(base.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(base.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        zi = zf.getinfo("predictions.csv")
    print("Validation:", flush=True)
    print(f"  rows={len(base):,}; type_counts={counts}", flush=True)
    print(f"  bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    print(f"  csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; zip_names={names}; uncompressed={zi.file_size:,}", flush=True)
    if len(base) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
        raise SystemExit("row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise SystemExit("content validation failed")
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
