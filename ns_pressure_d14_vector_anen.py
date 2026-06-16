from __future__ import annotations

import importlib.util
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
BASE_CSV = WORK / "predictions_anen_hybrid_v1_compact.csv"
OUT_CSV = WORK / "predictions_vector_anen_full_v1_compact.csv"
OUT_ZIP = WORK / "submission_vector_anen_full_v1_compact.zip"

REGION = "north_sea"
HORIZON = 14
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
HOURS = (0, 6, 12, 18)
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
YEARS = (2020, 2021)
K_GRID = (15, 30, 60)
SEASON_W_GRID = (0.0, 0.35, 0.75)
SAMPLE_PER_ANCHOR = 350

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
class GridAnenParams:
    k: int
    season_w: float
    half_width: float
    score: float


def cws(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    best = SOL.optimize_dir_halfwidth(y[ok], pred[ok], SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def angle_from_uv(u, v) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def circ_mean_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(np.radians(arr)).mean(), np.cos(np.radians(arr)).mean())) % 360.0)


def corrected_center_uv(
    target_u: float,
    target_v: float,
    hist_hu: np.ndarray,
    hist_hv: np.ndarray,
    hist_au: np.ndarray,
    hist_av: np.ndarray,
    hist_doy: np.ndarray,
    target_doy: int,
    k: int,
    season_w: float,
) -> float:
    ok = np.isfinite(hist_hu) & np.isfinite(hist_hv) & np.isfinite(hist_au) & np.isfinite(hist_av)
    if not np.isfinite(target_u) or not np.isfinite(target_v) or int(ok.sum()) < 3:
        return np.nan
    hu = hist_hu[ok]
    hv = hist_hv[ok]
    au = hist_au[ok]
    av = hist_av[ok]
    doy = hist_doy[ok].astype(float)
    uv_dist = np.sqrt((hu - target_u) ** 2 + (hv - target_v) ** 2) / 8.0
    ddoy = np.abs(doy - float(target_doy))
    ddoy = np.minimum(ddoy, 366.0 - ddoy) / 45.0
    dist = uv_dist + float(season_w) * ddoy
    take = np.argsort(dist)[: min(int(k), len(dist))]
    cu = float(target_u) + (au[take] - hu[take])
    cv = float(target_v) + (av[take] - hv[take])
    return circ_mean_deg(angle_from_uv(cu, cv))


def load_train_feature_columns() -> pd.DataFrame:
    cols = ["time", "latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        for hour in HOURS:
            cols.append(f"fcst_u_{level}_d10_h{hour}")
            cols.append(f"fcst_v_{level}_d10_h{hour}")
    feat = pd.read_parquet(FEATURES / f"train_{REGION}.parquet", columns=cols)
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = feat["latitude"].astype(float).round(2)
    feat["longitude"] = feat["longitude"].astype(float).round(2)
    feat["year"] = feat["time"].dt.year.astype("int16")
    feat["doy"] = feat["time"].dt.dayofyear.astype("int16")
    return feat


def load_pressure_actual() -> pd.DataFrame:
    cols = ["time", "latitude", "longitude"]
    for level in PRESSURE_LEVELS:
        cols.append(f"u_{level}")
        cols.append(f"v_{level}")
    actual = pd.read_parquet(DATA / "train" / f"reanalysis_pressure_{REGION}.parquet", columns=cols)
    actual["time"] = pd.to_datetime(actual["time"])
    actual["latitude"] = actual["latitude"].astype(float).round(2)
    actual["longitude"] = actual["longitude"].astype(float).round(2)
    return actual.set_index(["time", "latitude", "longitude"]).sort_index()


def attach_actual(feat: pd.DataFrame, actual: pd.DataFrame, level: str, hour: int) -> pd.DataFrame:
    out = feat[["time", "latitude", "longitude", "year", "doy", f"fcst_u_{level}_d10_h{hour}", f"fcst_v_{level}_d10_h{hour}"]].copy()
    out = out.rename(columns={f"fcst_u_{level}_d10_h{hour}": "hres_u", f"fcst_v_{level}_d10_h{hour}": "hres_v"})
    target_time = out["time"] + pd.Timedelta(days=HORIZON) + pd.Timedelta(hours=hour)
    keys = pd.MultiIndex.from_arrays(
        [target_time.values, out["latitude"].values, out["longitude"].values],
        names=["time", "latitude", "longitude"],
    )
    out["actual_u"] = actual[f"u_{level}"].reindex(keys).to_numpy(dtype="float64")
    out["actual_v"] = actual[f"v_{level}"].reindex(keys).to_numpy(dtype="float64")
    return out


def sampled_validation_queries(table: pd.DataFrame, level: str, hour: int) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        anchors = pd.to_datetime([f"{year}-{mmdd}" for mmdd in ANCHOR_MMDD])
        val = table[table["time"].isin(anchors)].copy()
        parts = []
        for _, part in val.groupby("time", sort=True):
            parts.append(part.sample(min(len(part), SAMPLE_PER_ANCHOR), random_state=year + hour))
        val = pd.concat(parts, ignore_index=True)
        val["val_year"] = year
        val["level"] = level
        val["hour"] = hour
        frames.append(val)
    return pd.concat(frames, ignore_index=True)


def backtest_level_hour(table: pd.DataFrame, level: str, hour: int) -> list[dict[str, object]]:
    queries = sampled_validation_queries(table, level, hour)
    hist_by_coord = {
        key: g.sort_values("time")
        for key, g in table.groupby(["latitude", "longitude"], sort=False)
    }
    rows = []
    actual_angle = angle_from_uv(queries["actual_u"].to_numpy(dtype="float64"), queries["actual_v"].to_numpy(dtype="float64"))
    for k in K_GRID:
        for season_w in SEASON_W_GRID:
            preds = []
            for _, q in queries.iterrows():
                hist = hist_by_coord.get((float(q["latitude"]), float(q["longitude"])))
                if hist is None:
                    preds.append(np.nan)
                    continue
                hist = hist[hist["year"].lt(int(q["val_year"]))]
                preds.append(
                    corrected_center_uv(
                        float(q["hres_u"]),
                        float(q["hres_v"]),
                        hist["hres_u"].to_numpy(dtype="float64"),
                        hist["hres_v"].to_numpy(dtype="float64"),
                        hist["actual_u"].to_numpy(dtype="float64"),
                        hist["actual_v"].to_numpy(dtype="float64"),
                        hist["doy"].to_numpy(dtype="int16"),
                        int(q["doy"]),
                        k,
                        season_w,
                    )
                )
            score, half_width = cws(actual_angle, np.asarray(preds, dtype="float64"))
            rows.append({"level": level, "hour": hour, "k": k, "season_w": season_w, "score": score, "half_width": half_width, "n": int(np.isfinite(actual_angle).sum())})
    return rows


def run_backtest(feat: pd.DataFrame, actual: pd.DataFrame) -> GridAnenParams:
    all_rows = []
    for level in PRESSURE_LEVELS:
        for hour in HOURS:
            print(f"Backtesting grid vector AnEn level={level} hour={hour}", flush=True)
            table = attach_actual(feat, actual, level, hour)
            all_rows.extend(backtest_level_hour(table, level, hour))
    raw = pd.DataFrame(all_rows)
    out_path = WORK / "ns_pressure_d14_vector_anen_backtest_by_level_hour.csv"
    raw.to_csv(out_path, index=False)
    pooled = (
        raw.groupby(["k", "season_w"], as_index=False)
        .agg(score=("score", "mean"), score_max=("score", "max"), half_width=("half_width", "median"))
        .sort_values(["score", "score_max"])
    )
    pooled_path = WORK / "ns_pressure_d14_vector_anen_backtest_summary.csv"
    pooled.to_csv(pooled_path, index=False)
    print(pooled.to_string(index=False), flush=True)
    best = pooled.iloc[0]
    params = GridAnenParams(k=int(best["k"]), season_w=float(best["season_w"]), half_width=float(best["half_width"]), score=float(best["score"]))
    print(f"Selected grid params: {params}", flush=True)
    return params


def load_inference_queries(df: pd.DataFrame, level: str, hour: int) -> pd.DataFrame:
    rows = []
    for window in range(1, 9):
        feat = pd.read_parquet(
            FEATURES / f"inference_window_{window}_{REGION}.parquet",
            columns=["time", "latitude", "longitude", f"fcst_u_{level}_d10_h{hour}", f"fcst_v_{level}_d10_h{hour}"],
        )
        feat["time"] = pd.to_datetime(feat["time"])
        feat["latitude"] = feat["latitude"].astype(float).round(2)
        feat["longitude"] = feat["longitude"].astype(float).round(2)
        feat["window"] = window
        feat["doy"] = feat["time"].dt.dayofyear.astype("int16")
        feat = feat.rename(columns={f"fcst_u_{level}_d10_h{hour}": "hres_u", f"fcst_v_{level}_d10_h{hour}": "hres_v"})
        rows.append(feat[["window", "time", "latitude", "longitude", "doy", "hres_u", "hres_v"]])
    queries = pd.concat(rows, ignore_index=True)
    key = df[
        df["type"].eq("grid")
        & df["region"].eq(REGION)
        & df["horizon"].eq(HORIZON)
        & df["hour"].eq(hour)
        & df["level"].eq(level)
    ].reset_index()[["index", "window", "latitude", "longitude"]]
    queries = queries.merge(key, on=["window", "latitude", "longitude"], how="left", validate="one_to_one")
    if queries["index"].isna().any():
        raise SystemExit(f"missing output indices for level={level} hour={hour}: {int(queries['index'].isna().sum())}")
    return queries


def patch_inference(params: GridAnenParams, feat: pd.DataFrame, actual: pd.DataFrame) -> None:
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[COLS].copy()
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    df["station"] = df["station"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    changed = 0
    for level in PRESSURE_LEVELS:
        for hour in HOURS:
            print(f"Patching grid vector AnEn level={level} hour={hour}", flush=True)
            table = attach_actual(feat, actual, level, hour)
            hist_by_coord = {
                key: g[g["actual_u"].notna() & g["actual_v"].notna()].sort_values("time")
                for key, g in table.groupby(["latitude", "longitude"], sort=False)
            }
            queries = load_inference_queries(df, level, hour)
            for _, q in queries.iterrows():
                hist = hist_by_coord.get((float(q["latitude"]), float(q["longitude"])))
                if hist is None or len(hist) < 3:
                    continue
                center = corrected_center_uv(
                    float(q["hres_u"]),
                    float(q["hres_v"]),
                    hist["hres_u"].to_numpy(dtype="float64"),
                    hist["hres_v"].to_numpy(dtype="float64"),
                    hist["actual_u"].to_numpy(dtype="float64"),
                    hist["actual_v"].to_numpy(dtype="float64"),
                    hist["doy"].to_numpy(dtype="int16"),
                    int(q["doy"]),
                    params.k,
                    params.season_w,
                )
                if not np.isfinite(center):
                    continue
                idx = int(q["index"])
                df.at[idx, "dir_50"] = center % 360.0
                df.at[idx, "dir_05"] = (center - params.half_width) % 360.0
                df.at[idx, "dir_95"] = (center + params.half_width) % 360.0
                changed += 1
    print(f"Patched NS pressure d14 vector-AnEn rows: {changed:,}", flush=True)

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
    print("Validation:", flush=True)
    print(f"  rows={len(df):,}; type_counts={counts}", flush=True)
    print(f"  bad_speed={bad_speed}; bad_dir={bad_dir}; missing={missing}; grid_dup={grid_dup}; station_dup={station_dup}", flush=True)
    if len(df) != 3_448_800 or counts.get("grid") != 3_447_360 or counts.get("station") != 1_440:
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
    print("Loading North Sea pressure/HRES training tables", flush=True)
    feat = load_train_feature_columns()
    actual = load_pressure_actual()
    params = run_backtest(feat, actual)
    patch_inference(params, feat, actual)


if __name__ == "__main__":
    main()
