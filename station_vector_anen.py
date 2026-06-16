from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"
BASE_CSV = WORK / "predictions_public_positive_fullrefit_hybrid_compact.csv"
OUT_CSV = WORK / "predictions_station_vector_anen_v1_compact.csv"
OUT_ZIP = WORK / "submission_station_vector_anen_v1_compact.zip"

REGIONS = ("north_sea", "east_china_sea")
HORIZONS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
YEARS = (2020, 2021)
K_GRID = (15, 30, 60)
SEASON_W_GRID = (0.0, 0.35, 0.75)
PATCH_BLOCKS = {
    ("north_sea", 7),
    ("north_sea", 14),
    ("east_china_sea", 1),
}

COLS = [
    "type", "window", "region", "latitude", "longitude", "station",
    "horizon", "hour", "level", "q05", "q50", "q95",
    "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def load_solution_module():
    path = ROOT / "sea_winds_solution_ephemeral_v6_pressure_speed.py"
    spec = importlib.util.spec_from_file_location("sea_winds_solution_v6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SOL = load_solution_module()


@dataclass(frozen=True)
class AnenParams:
    k: int
    season_w: float
    half_width: float


def circular_distance(a, b) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def circ_mean_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(np.radians(arr)).mean(), np.cos(np.radians(arr)).mean())) % 360.0)


def corrected_center(target_hres: float, hist_hres: np.ndarray, hist_actual: np.ndarray, k: int, season_w: float, target_doy: int, hist_doy: np.ndarray) -> float:
    ok = np.isfinite(hist_hres) & np.isfinite(hist_actual)
    if not np.isfinite(target_hres) or int(ok.sum()) < 3:
        return np.nan
    hh = hist_hres[ok]
    yy = hist_actual[ok]
    ddoy = np.abs(hist_doy[ok].astype(float) - float(target_doy))
    ddoy = np.minimum(ddoy, 366.0 - ddoy) / 45.0
    dist = circular_distance(hh, target_hres) / 45.0 + float(season_w) * ddoy
    take = np.argsort(dist)[: min(int(k), len(dist))]
    hist_vec_x = np.cos(np.radians(hh[take]))
    hist_vec_y = np.sin(np.radians(hh[take]))
    act_vec_x = np.cos(np.radians(yy[take]))
    act_vec_y = np.sin(np.radians(yy[take]))
    target_x = np.cos(np.radians(float(target_hres)))
    target_y = np.sin(np.radians(float(target_hres)))
    corrected_x = target_x + (act_vec_x - hist_vec_x)
    corrected_y = target_y + (act_vec_y - hist_vec_y)
    corrected = np.degrees(np.arctan2(corrected_y, corrected_x)) % 360.0
    return circ_mean_deg(corrected)


def cws(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    best = SOL.optimize_dir_halfwidth(y[ok], pred[ok], SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def read_anchor(window: int) -> pd.Timestamp:
    meta = json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text())
    return pd.Timestamp(meta["context_end"])


def hres_col(horizon: int, hour: int) -> str:
    lead = horizon if horizon in (1, 7) else 10
    return f"fcst_dir_d{lead}_h{hour}"


def load_station_tables(region: str, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["time", "latitude", "longitude"]
    for horizon in HORIZONS:
        for hour in HOURS:
            cols.append(hres_col(horizon, hour))
    feat = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=sorted(set(cols)))
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = feat["latitude"].astype(float).round(2)
    feat["longitude"] = feat["longitude"].astype(float).round(2)

    obs = pd.read_parquet(DATA / "train" / f"stations_{region}_6h.parquet", columns=["time", "station", "direction"])
    obs["time"] = pd.to_datetime(obs["time"])
    obs["station"] = obs["station"].astype(str)
    obs["direction"] = pd.to_numeric(obs["direction"], errors="coerce") % 360.0

    tables = {}
    for _, row in meta[meta["region"].eq(region)].iterrows():
        station = str(row["station"])
        lat = round(float(row["nearest_grid_lat"]), 2)
        lon = round(float(row["nearest_grid_lon"]), 2)
        sub = feat[feat["latitude"].eq(lat) & feat["longitude"].eq(lon)].copy()
        sub["station"] = station
        sub["doy"] = sub["time"].dt.dayofyear.astype("int16")
        for horizon in HORIZONS:
            for hour in HOURS:
                tgt = sub[["time", "station", "doy", hres_col(horizon, hour)]].copy()
                tgt["target_time"] = tgt["time"] + pd.Timedelta(days=horizon) + pd.Timedelta(hours=hour)
                tgt = tgt.merge(obs, left_on=["station", "target_time"], right_on=["station", "time"], how="left", suffixes=("", "_obs"))
                tgt = tgt.rename(columns={hres_col(horizon, hour): "hres_dir", "direction": "actual_dir"})
                tables[(station, horizon, hour)] = tgt[["time", "station", "doy", "hres_dir", "actual_dir"]].copy()
    return tables


def predict_backtest_region(region: str, meta: pd.DataFrame) -> pd.DataFrame:
    tables = load_station_tables(region, meta)
    rows = []
    for horizon in HORIZONS:
        for k in K_GRID:
            for season_w in SEASON_W_GRID:
                y_all = []
                p_all = []
                for year in YEARS:
                    anchors = pd.to_datetime([f"{year}-{mmdd}" for mmdd in ANCHOR_MMDD])
                    for _, st in meta[meta["region"].eq(region)].iterrows():
                        station = str(st["station"])
                        for hour in HOURS:
                            table = tables[(station, horizon, hour)]
                            hist = table[table["time"].dt.year.lt(year)]
                            hist_hres = pd.to_numeric(hist["hres_dir"], errors="coerce").to_numpy(dtype="float64") % 360.0
                            hist_actual = pd.to_numeric(hist["actual_dir"], errors="coerce").to_numpy(dtype="float64") % 360.0
                            hist_doy = hist["doy"].to_numpy(dtype="int16")
                            val = table[table["time"].isin(anchors)]
                            for _, row in val.iterrows():
                                y = float(row["actual_dir"]) if np.isfinite(row["actual_dir"]) else np.nan
                                pred = corrected_center(
                                    float(row["hres_dir"]),
                                    hist_hres,
                                    hist_actual,
                                    k,
                                    season_w,
                                    int(row["doy"]),
                                    hist_doy,
                                )
                                y_all.append(y)
                                p_all.append(pred)
                y_arr = np.asarray(y_all, dtype="float64")
                p_arr = np.asarray(p_all, dtype="float64")
                score, half_width = cws(y_arr, p_arr)
                rows.append(
                    {
                        "region": region,
                        "horizon": horizon,
                        "k": k,
                        "season_w": season_w,
                        "score": score,
                        "half_width": half_width,
                        "n": int(np.isfinite(y_arr).sum()),
                    }
                )
    return pd.DataFrame(rows)


def select_params(summary: pd.DataFrame) -> dict[tuple[str, int], AnenParams]:
    params = {}
    for (region, horizon), g in summary.groupby(["region", "horizon"], sort=False):
        if (region, int(horizon)) not in PATCH_BLOCKS:
            continue
        best = g.sort_values(["score", "k"]).iloc[0]
        params[(region, int(horizon))] = AnenParams(
            k=int(best["k"]),
            season_w=float(best["season_w"]),
            half_width=float(best["half_width"]),
        )
    return params


def load_inference_hres(region: str, window: int, horizon: int, hour: int, meta: pd.DataFrame) -> dict[str, float]:
    col = hres_col(horizon, hour)
    feat = pd.read_parquet(FEATURES / f"inference_window_{window}_{region}.parquet", columns=["latitude", "longitude", col])
    feat["latitude"] = feat["latitude"].astype(float).round(2)
    feat["longitude"] = feat["longitude"].astype(float).round(2)
    lookup = feat.set_index(["latitude", "longitude"])[col]
    out = {}
    for _, row in meta[meta["region"].eq(region)].iterrows():
        station = str(row["station"])
        key = (round(float(row["nearest_grid_lat"]), 2), round(float(row["nearest_grid_lon"]), 2))
        try:
            out[station] = float(lookup.loc[key]) % 360.0
        except KeyError:
            out[station] = np.nan
    return out


def build_candidate(params: dict[tuple[str, int], AnenParams], meta: pd.DataFrame) -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[COLS].copy()
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    df["station"] = df["station"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")

    train_tables = {region: load_station_tables(region, meta) for region in REGIONS}
    counts = {}
    for (region, horizon), par in params.items():
        counts[(region, horizon)] = 0
        for window in range(1, 9):
            anchor = read_anchor(window)
            for hour in HOURS:
                hres_now = load_inference_hres(region, window, horizon, hour, meta)
                mask = (
                    df["type"].eq("station")
                    & df["region"].eq(region)
                    & df["window"].eq(window)
                    & df["horizon"].eq(horizon)
                    & df["hour"].eq(hour)
                )
                for idx in df.index[mask]:
                    station = str(df.at[idx, "station"])
                    table = train_tables[region][(station, horizon, hour)]
                    hist = table[table["actual_dir"].notna()].copy()
                    hist_hres = pd.to_numeric(hist["hres_dir"], errors="coerce").to_numpy(dtype="float64") % 360.0
                    hist_actual = pd.to_numeric(hist["actual_dir"], errors="coerce").to_numpy(dtype="float64") % 360.0
                    hist_doy = hist["doy"].to_numpy(dtype="int16")
                    center = corrected_center(
                        hres_now.get(station, np.nan),
                        hist_hres,
                        hist_actual,
                        par.k,
                        par.season_w,
                        int(anchor.dayofyear),
                        hist_doy,
                    )
                    if not np.isfinite(center):
                        continue
                    df.at[idx, "dir_50"] = center % 360.0
                    df.at[idx, "dir_05"] = (center - par.half_width) % 360.0
                    df.at[idx, "dir_95"] = (center + par.half_width) % 360.0
                    counts[(region, horizon)] += 1
    print(f"Patch counts: {counts}", flush=True)

    for c in SPEED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).round(2)
    df["q05"] = df[["q05", "q50"]].min(axis=1).round(2)
    df["q95"] = df[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)
    grid = df["type"].eq("grid")
    type_counts = df["type"].value_counts(dropna=False).to_dict()
    bad_speed = int(((df["q05"] > df["q50"]) | (df["q50"] > df["q95"]) | (df[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((df[DIR_COLS] < 0) | (df[DIR_COLS] >= 360) | df[DIR_COLS].isna()).any(axis=1).sum())
    missing = int(df[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    grid_dup = int(df.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(df.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    print("Validation:", flush=True)
    print(f"  rows={len(df):,}; type_counts={type_counts}", flush=True)
    print(f"  bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    if len(df) != 3_448_800 or type_counts.get("grid") != 3_447_360 or type_counts.get("station") != 1_440:
        raise SystemExit("row/type count validation failed")
    if bad_speed or bad_dir or missing or grid_dup or station_dup:
        raise SystemExit("content validation failed")

    print(f"Writing {OUT_CSV}", flush=True)
    df[COLS].to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    print(f"csv_size={OUT_CSV.stat().st_size:,}; zip_size={OUT_ZIP.stat().st_size:,}; names={names}; uncompressed={info.file_size:,}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


def main() -> None:
    meta = pd.read_csv(DATA / "scoring" / "station_metadata.csv")
    rows = []
    for region in REGIONS:
        print(f"Backtesting station vector AnEn: {region}", flush=True)
        rows.append(predict_backtest_region(region, meta))
    summary = pd.concat(rows, ignore_index=True)
    out_path = WORK / "station_vector_anen_backtest.csv"
    summary.to_csv(out_path, index=False)
    best = summary.sort_values(["region", "horizon", "score"]).groupby(["region", "horizon"]).head(3)
    print(best.to_string(index=False), flush=True)
    params = select_params(summary)
    print("Selected params:", params, flush=True)
    build_candidate(params, meta)


if __name__ == "__main__":
    main()
