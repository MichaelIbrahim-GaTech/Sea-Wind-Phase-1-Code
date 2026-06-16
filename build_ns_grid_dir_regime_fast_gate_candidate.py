from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import hres_mos_residual_branch as HM
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

BASE_CSV = WORK / "pred_ns_p7dir_mosres.csv"
OUT_CSV = WORK / "pred_ns_grid_regime_anen.csv"
OUT_ZIP = WORK / "sub_ns_grid_anen.zip"
MANIFEST = WORK / "manifest_ns_grid_regime_anen.json"
CV_BY_FOLD = WORK / "cv_ns_grid_regime_anen_by_fold.csv"
CV_SUMMARY = WORK / "cv_ns_grid_regime_anen_summary.csv"

REGION = "north_sea"
HOURS = (0, 6, 12, 18)
SURFACE_LEVELS = ("10m", "100m")
PRESSURE_LEVELS = ("1000", "925", "850", "700", "500")
ANCHOR_MMDD = ("01-14", "02-25", "04-08", "05-20", "07-01", "08-12", "09-23", "11-04")
VAL_YEARS = (2020, 2021)

# Rank-damaging grouped North Sea grid-direction blocks. The builder gates each
# block separately, so a weak block cannot contaminate a stronger one.
BLOCKS = (
    ("surface", 7),
    ("surface", 14),
    ("pressure", 14),
)

REGIME_FEATURES = (
    "msl",
    "ws10",
    "ws100",
    "wind_shear",
    "t2m",
    "sst",
    "z700",
    "cape",
    "ns_pressure_gradient",
)
ANALOG_WINDOWS = (45, 90)
ANALOG_KS = (5, 20)
BLEND_WEIGHTS_ANALOG = (0.25, 0.50, 0.75)
SAMPLE_PER_ANCHOR = 512
RANDOM_SEED = 20260610

PUBLIC_CURRENT = {
    ("surface", 7): 298.5943,
    ("surface", 14): 340.4365,
    ("pressure", 14): 330.5092,
}
HISTORICAL_BASELINE_REF = {
    ("surface", 7): 318.2809410317545,
    ("surface", 14): 328.45183243269076,
    ("pressure", 14): 333.99884030914274,
}

MIN_MEAN_GAIN = 8.0
MIN_PUBLIC_MARGIN = 6.0
MAX_WORST_WORSE = 0.0
MAX_INFERENCE_CENTER_SHIFT_MEAN = 75.0
MAX_INFERENCE_CENTER_SHIFT_P90 = 150.0

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS
KEYS = E2E.KEYS


@dataclass(frozen=True)
class SelectedBlock:
    group: str
    horizon: int
    candidate: str
    half_width: float
    score_mean: float
    score_max: float
    baseline_ref: float
    public_current: float


@dataclass
class SurfaceTargetStore:
    n_grid: int
    time_to_idx: dict[pd.Timestamp, int]
    targets: dict[str, np.ndarray]


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


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP, MANIFEST, CV_BY_FOLD, CV_SUMMARY):
        if path.exists():
            path.unlink()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def circular_mean_deg(values: Iterable[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    arr = np.vstack([np.asarray(v, dtype="float64") % 360.0 for v in values])
    valid = np.isfinite(arr)
    rad = np.deg2rad(np.where(valid, arr, 0.0))
    if weights is None:
        w = valid.astype("float64")
    else:
        w = np.asarray(weights, dtype="float64")[:, None] * valid.astype("float64")
    sin = np.sum(np.sin(rad) * w, axis=0)
    cos = np.sum(np.cos(rad) * w, axis=0)
    out = np.degrees(np.arctan2(sin, cos)) % 360.0
    out[(np.abs(sin) + np.abs(cos)) <= 1e-12] = np.nan
    return out


def blend_direction_deg(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    a = np.asarray(a, dtype="float64") % 360.0
    b = np.asarray(b, dtype="float64") % 360.0
    ar = np.deg2rad(a)
    br = np.deg2rad(b)
    x = (1.0 - weight_b) * np.cos(ar) + weight_b * np.cos(br)
    y = (1.0 - weight_b) * np.sin(ar) + weight_b * np.sin(br)
    out = np.degrees(np.arctan2(y, x)) % 360.0
    out[~np.isfinite(a) | ~np.isfinite(b)] = np.nan
    return out


def cws(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype="float64")
    center = np.asarray(center, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(center)
    if not bool(ok.any()):
        return float("nan"), float("nan")
    score, hw = HM.best_direction_width(y[ok], center[ok])
    return float(score), float(hw)


def target_levels(group: str) -> tuple[str, ...]:
    return SURFACE_LEVELS if group == "surface" else PRESSURE_LEVELS


def lead_for_horizon(horizon: int) -> int:
    return horizon if horizon == 7 else 10


def train_feature_columns() -> list[str]:
    cols = {"time", "latitude", "longitude"}
    cols.update(REGIME_FEATURES)
    for horizon in (7, 14):
        lead = lead_for_horizon(horizon)
        for hour in HOURS:
            cols.add(f"fcst_dir_d{lead}_h{hour}")
            cols.add(f"dir_d{horizon}_h{hour}")
            for level in PRESSURE_LEVELS:
                cols.add(f"fcst_u_{level}_d{lead}_h{hour}")
                cols.add(f"fcst_v_{level}_d{lead}_h{hour}")
    return sorted(cols)


def inference_feature_columns() -> list[str]:
    cols = {"time", "latitude", "longitude"}
    cols.update(REGIME_FEATURES)
    for horizon in (7, 14):
        lead = lead_for_horizon(horizon)
        for hour in HOURS:
            cols.add(f"fcst_dir_d{lead}_h{hour}")
            for level in PRESSURE_LEVELS:
                cols.add(f"fcst_u_{level}_d{lead}_h{hour}")
                cols.add(f"fcst_v_{level}_d{lead}_h{hour}")
    return sorted(cols)


def load_train_features() -> pd.DataFrame:
    print("Loading official North Sea train features", flush=True)
    df = pd.read_parquet(FEATURES / f"train_{REGION}.parquet", columns=train_feature_columns())
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(2)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(2)
    df["year"] = df["time"].dt.year.astype("int16")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    print(f"  train feature rows={len(df):,} cols={len(df.columns)}", flush=True)
    return df


def load_inference_features(window: int, grid: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES / f"inference_window_{window}_{REGION}.parquet", columns=inference_feature_columns())
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(2)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(2)
    df["year"] = df["time"].dt.year.astype("int16")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    return attach_grid_idx(df, grid)


def build_grid(feat: pd.DataFrame) -> pd.DataFrame:
    grid = feat[["latitude", "longitude"]].drop_duplicates().sort_values(["latitude", "longitude"], kind="mergesort")
    grid = grid.reset_index(drop=True)
    grid["grid_idx"] = np.arange(len(grid), dtype="int32")
    if len(grid) != 2565:
        raise SystemExit(f"unexpected North Sea grid size: {len(grid)}")
    return grid


def attach_grid_idx(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(grid, on=["latitude", "longitude"], how="left", sort=False, validate="many_to_one")
    if out["grid_idx"].isna().any():
        bad = out.loc[out["grid_idx"].isna(), ["latitude", "longitude"]].drop_duplicates().head()
        raise SystemExit(f"missing grid_idx for coordinates:\n{bad}")
    out["grid_idx"] = out["grid_idx"].astype("int32")
    return out


def pressure_index_map(grid: pd.DataFrame, cube: HM.CubeStore) -> np.ndarray:
    cube_grid = cube.latlon.copy()
    cube_grid["latitude"] = pd.to_numeric(cube_grid["latitude"], errors="coerce").round(2)
    cube_grid["longitude"] = pd.to_numeric(cube_grid["longitude"], errors="coerce").round(2)
    cube_grid = cube_grid.reset_index().rename(columns={"index": "cube_idx"})
    merged = grid.merge(cube_grid, on=["latitude", "longitude"], how="left", sort=False, validate="one_to_one")
    if merged["cube_idx"].isna().any():
        raise SystemExit("pressure cube grid does not align with feature grid")
    return merged.sort_values("grid_idx", kind="mergesort")["cube_idx"].to_numpy(dtype="int32")


def build_surface_store(feat: pd.DataFrame, grid: pd.DataFrame) -> SurfaceTargetStore:
    print("Building compact official surface target store from train labels", flush=True)
    columns = [f"dir_d{horizon}_h{hour}" for horizon in (7, 14) for hour in HOURS]
    ordered = attach_grid_idx(feat[["time", "latitude", "longitude"] + columns].copy(), grid)
    ordered = ordered.sort_values(["time", "grid_idx"], kind="mergesort").reset_index(drop=True)
    times = pd.Series(ordered["time"].unique()).sort_values().reset_index(drop=True)
    n_grid = len(grid)
    if len(ordered) != len(times) * n_grid:
        raise SystemExit("train feature grid is not rectangular; refusing surface target store")
    targets = {}
    for col in columns:
        targets[col] = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype="float32").reshape(len(times), n_grid)
    return SurfaceTargetStore(n_grid=n_grid, time_to_idx={pd.Timestamp(t): i for i, t in enumerate(times)}, targets=targets)


def load_pressure_cube() -> tuple[HM.CubeStore, np.ndarray]:
    print("Loading official North Sea pressure reanalysis cube", flush=True)
    cube = HM.load_cube(REGION, "pressure")
    print(f"  pressure cube times={len(cube.time_to_idx):,} grid={cube.n_grid:,}", flush=True)
    return cube, np.empty(0, dtype="int32")


def feature_daily_means(feat: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REGIME_FEATURES if c in feat.columns]
    means = feat.groupby("time", sort=True)[cols].mean(numeric_only=True).reset_index()
    means["year"] = means["time"].dt.year.astype("int16")
    means["doy"] = means["time"].dt.dayofyear.astype("int16")
    return means


def regime_scale(means: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cols = [c for c in REGIME_FEATURES if c in means.columns]
    mu = means[cols].mean(numeric_only=True)
    sigma = means[cols].std(numeric_only=True).replace(0, np.nan).fillna(1.0)
    return mu, sigma


def doy_distance(a: pd.Series, b: int) -> pd.Series:
    d = (a.astype(int) - int(b)).abs()
    return np.minimum(d, 366 - d)


def select_regime_analogs(
    means: pd.DataFrame,
    query_row: pd.Series,
    train_year_max: int,
    window_days: int,
    k: int,
    mu: pd.Series,
    sigma: pd.Series,
) -> pd.DataFrame:
    cols = list(mu.index)
    pool = means[
        means["year"].le(int(train_year_max))
        & (doy_distance(means["doy"], int(query_row["doy"])) <= int(window_days))
    ].copy()
    if pool.empty:
        pool = means[means["year"].le(int(train_year_max))].copy()
    q = ((query_row[cols].astype("float64") - mu) / sigma).to_numpy(dtype="float64")
    x = ((pool[cols].astype("float64") - mu) / sigma).to_numpy(dtype="float64")
    dist = np.sqrt(np.nanmean((x - q[None, :]) ** 2, axis=1))
    pool["analog_dist"] = dist
    return pool.sort_values("analog_dist", kind="mergesort").head(int(k)).reset_index(drop=True)


def hres_center(feat: pd.DataFrame, group: str, horizon: int, level: str, hour: int) -> np.ndarray:
    lead = lead_for_horizon(horizon)
    if group == "surface":
        return pd.to_numeric(feat[f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0
    u = pd.to_numeric(feat[f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(feat[f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def surface_target(store: SurfaceTargetStore, origin_time: pd.Timestamp, grid_idx: np.ndarray, horizon: int, hour: int) -> np.ndarray:
    time_idx = store.time_to_idx.get(pd.Timestamp(origin_time))
    if time_idx is None:
        return np.full(len(grid_idx), np.nan)
    return store.targets[f"dir_d{horizon}_h{hour}"][time_idx, grid_idx].astype("float64") % 360.0


def pressure_target(
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
    target_time: pd.Timestamp,
    grid_idx: np.ndarray,
    level: str,
) -> np.ndarray:
    time_idx = cube.time_to_idx.get(pd.Timestamp(target_time))
    if time_idx is None:
        return np.full(len(grid_idx), np.nan)
    cube_idx = cube_idx_map[grid_idx]
    u = cube.u[level][time_idx, cube_idx].astype("float64")
    v = cube.v[level][time_idx, cube_idx].astype("float64")
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def target_for(
    group: str,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
    origin_time: pd.Timestamp,
    grid_idx: np.ndarray,
    level: str,
    horizon: int,
    hour: int,
) -> np.ndarray:
    if group == "surface":
        return surface_target(store, origin_time, grid_idx, horizon, hour)
    target_time = pd.Timestamp(origin_time) + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    return pressure_target(cube, cube_idx_map, target_time, grid_idx, level)


def analog_center(
    group: str,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
    analogs: pd.DataFrame,
    grid_idx: np.ndarray,
    level: str,
    horizon: int,
    hour: int,
) -> np.ndarray:
    parts = []
    weights = []
    for _, r in analogs.iterrows():
        vals = target_for(
            group,
            store,
            cube,
            cube_idx_map,
            pd.Timestamp(r["time"]),
            grid_idx,
            level,
            horizon,
            hour,
        )
        parts.append(vals)
        weights.append(1.0 / (float(r["analog_dist"]) + 0.05))
    if not parts:
        return np.full(len(grid_idx), np.nan)
    return circular_mean_deg(parts, np.asarray(weights, dtype="float64"))


def sample_anchor_rows(feat: pd.DataFrame, anchors: pd.DatetimeIndex, val_year: int) -> pd.DataFrame:
    rows = feat[feat["time"].isin(anchors)].copy()
    parts = []
    rng_seed = RANDOM_SEED + int(val_year)
    for _, part in rows.groupby("time", sort=True):
        parts.append(part.sample(min(len(part), SAMPLE_PER_ANCHOR), random_state=rng_seed))
    if len(parts) != len(anchors):
        raise SystemExit(f"missing one or more CV anchors for {val_year}: expected {len(anchors)} got {len(parts)}")
    return pd.concat(parts, ignore_index=True)


def candidate_scores(y_cat: np.ndarray, hres_cat: np.ndarray, analog_parts: dict[str, list[np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    hres_score, hres_hw = cws(y_cat, hres_cat)
    rows.append({"candidate": "hres", "score": hres_score, "half_width": hres_hw})
    for candidate, parts in analog_parts.items():
        analog_cat = np.concatenate(parts)
        score, hw = cws(y_cat, analog_cat)
        rows.append({"candidate": candidate, "score": score, "half_width": hw})
        for weight in BLEND_WEIGHTS_ANALOG:
            blended = blend_direction_deg(hres_cat, analog_cat, weight)
            b_score, b_hw = cws(y_cat, blended)
            rows.append(
                {
                    "candidate": f"blend_hres_{candidate}_{weight:.2f}",
                    "score": b_score,
                    "half_width": b_hw,
                }
            )
    return rows


def evaluate_block(
    feat: pd.DataFrame,
    means: pd.DataFrame,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
    group: str,
    horizon: int,
    val_year: int,
) -> list[dict[str, object]]:
    anchors = pd.to_datetime([f"{val_year}-{mmdd}" for mmdd in ANCHOR_MMDD])
    eval_rows = sample_anchor_rows(feat, anchors, val_year)
    levels = target_levels(group)
    mu, sigma = regime_scale(means[means["year"].lt(int(val_year))])
    anchor_query = means.set_index("time", drop=False)

    y_parts = []
    hres_parts = []
    analog_parts: dict[str, list[np.ndarray]] = {}
    for window_days in ANALOG_WINDOWS:
        for k in ANALOG_KS:
            analog_parts[f"analog_w{window_days}_k{k}"] = []

    for anchor in anchors:
        anchor_rows = eval_rows[eval_rows["time"].eq(anchor)].reset_index(drop=True)
        grid_idx = anchor_rows["grid_idx"].to_numpy(dtype="int32")
        query = anchor_query.loc[pd.Timestamp(anchor)]
        analog_cache = {
            (window_days, k): select_regime_analogs(
                means,
                query,
                train_year_max=val_year - 1,
                window_days=window_days,
                k=k,
                mu=mu,
                sigma=sigma,
            )
            for window_days in ANALOG_WINDOWS
            for k in ANALOG_KS
        }
        for hour in HOURS:
            for level in levels:
                y = target_for(group, store, cube, cube_idx_map, anchor, grid_idx, level, horizon, hour)
                h = hres_center(anchor_rows, group, horizon, level, hour)
                y_parts.append(y)
                hres_parts.append(h)
                for key, parts in analog_parts.items():
                    _, wtxt, ktxt = key.split("_")
                    window_days = int(wtxt[1:])
                    k = int(ktxt[1:])
                    parts.append(
                        analog_center(
                            group,
                            store,
                            cube,
                            cube_idx_map,
                            analog_cache[(window_days, k)],
                            grid_idx,
                            level,
                            horizon,
                            hour,
                        )
                    )

    y_cat = np.concatenate(y_parts)
    hres_cat = np.concatenate(hres_parts)
    rows = []
    for score_row in candidate_scores(y_cat, hres_cat, analog_parts):
        rows.append(
            {
                "region": REGION,
                "group": group,
                "horizon": horizon,
                "val_year": val_year,
                "candidate": score_row["candidate"],
                "score": score_row["score"],
                "half_width": score_row["half_width"],
                "n": int(np.isfinite(y_cat).sum()),
            }
        )
    return rows


def summarize_cv(by_fold: pd.DataFrame) -> pd.DataFrame:
    summary = (
        by_fold.groupby(["region", "group", "horizon", "candidate"], as_index=False)
        .agg(score_mean=("score", "mean"), score_max=("score", "max"), half_width_mean=("half_width", "mean"))
    )
    records = []
    for _, row in summary.iterrows():
        key = (str(row["group"]), int(row["horizon"]))
        baseline_ref = float(HISTORICAL_BASELINE_REF[key])
        public_current = float(PUBLIC_CURRENT[key])
        mean_gain = baseline_ref - float(row["score_mean"])
        max_gain = baseline_ref - float(row["score_max"])
        public_margin = public_current - float(row["score_mean"])
        gate = (
            str(row["candidate"]) != "hres"
            and mean_gain >= MIN_MEAN_GAIN
            and max_gain >= MAX_WORST_WORSE
            and public_margin >= MIN_PUBLIC_MARGIN
        )
        out = row.to_dict()
        out.update(
            {
                "baseline_ref": baseline_ref,
                "public_current": public_current,
                "mean_gain_vs_baseline_ref": mean_gain,
                "max_gain_vs_baseline_ref": max_gain,
                "public_margin": public_margin,
                "gate_passed": bool(gate),
            }
        )
        records.append(out)
    return pd.DataFrame(records).sort_values(
        ["gate_passed", "group", "horizon", "score_mean"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def selected_blocks(summary: pd.DataFrame) -> list[SelectedBlock]:
    out = []
    for group, horizon in BLOCKS:
        sub = summary[summary["group"].eq(group) & summary["horizon"].astype(int).eq(int(horizon))]
        sub = sub[sub["gate_passed"].astype(bool)].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["score_mean", "score_max"], kind="mergesort").iloc[0]
        out.append(
            SelectedBlock(
                group=group,
                horizon=int(horizon),
                candidate=str(best["candidate"]),
                half_width=float(best["half_width_mean"]),
                score_mean=float(best["score_mean"]),
                score_max=float(best["score_max"]),
                baseline_ref=float(best["baseline_ref"]),
                public_current=float(best["public_current"]),
            )
        )
    return out


def parse_candidate(candidate: str) -> tuple[str, int, int, float]:
    if candidate == "hres":
        return "hres", 0, 0, 0.0
    parts = candidate.split("_")
    if candidate.startswith("blend_hres_"):
        return "blend", int(parts[3][1:]), int(parts[4][1:]), float(parts[5])
    return "analog", int(parts[1][1:]), int(parts[2][1:]), 1.0


def candidate_center_for_block(
    feat: pd.DataFrame,
    means: pd.DataFrame,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
    block: SelectedBlock,
    window: int,
) -> pd.DataFrame:
    levels = target_levels(block.group)
    grid_idx = feat["grid_idx"].to_numpy(dtype="int32")
    coords = feat[["latitude", "longitude"]].reset_index(drop=True)
    query = feature_daily_means(feat).iloc[0]
    mu, sigma = regime_scale(means[means["year"].le(2021)])
    mode, window_days, k, weight = parse_candidate(block.candidate)

    analogs = None
    if mode in {"analog", "blend"}:
        analogs = select_regime_analogs(
            means,
            query,
            train_year_max=2021,
            window_days=window_days,
            k=k,
            mu=mu,
            sigma=sigma,
        )

    rows = []
    for hour in HOURS:
        for level in levels:
            h = hres_center(feat, block.group, block.horizon, level, hour)
            if mode == "hres":
                center = h
            else:
                assert analogs is not None
                analog = analog_center(block.group, store, cube, cube_idx_map, analogs, grid_idx, level, block.horizon, hour)
                center = analog if mode == "analog" else blend_direction_deg(h, analog, weight)
            part = coords.copy()
            part["window"] = int(window)
            part["region"] = REGION
            part["horizon"] = int(block.horizon)
            part["hour"] = int(hour)
            part["level"] = level
            part["center"] = center
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    for c in SPEED_COLS + DIR_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_blocks(
    base: pd.DataFrame,
    blocks: list[SelectedBlock],
    grid: pd.DataFrame,
    means: pd.DataFrame,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    patched = base.copy()
    patch_counts: dict[str, object] = {}
    stability = []
    for block in blocks:
        print(f"Applying accepted block {block.group}/d{block.horizon}: {block.candidate}", flush=True)
        all_patches = []
        for window in range(1, 9):
            feat = load_inference_features(window, grid)
            all_patches.append(candidate_center_for_block(feat, means, store, cube, cube_idx_map, block, window))
        patch = pd.concat(all_patches, ignore_index=True)
        expected = 8 * len(target_levels(block.group)) * len(HOURS) * 2565
        if len(patch) != expected:
            raise SystemExit(f"unexpected patch rows for {block.group}/d{block.horizon}: {len(patch):,} != {expected:,}")
        target = (
            patched["type"].eq("grid")
            & patched["region"].eq(REGION)
            & patched["horizon"].eq(block.horizon)
            & patched["level"].isin(target_levels(block.group))
        )
        lookup = patched.loc[target].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level", "dir_50"]]
        merged = lookup.merge(
            patch,
            on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
            how="left",
            validate="one_to_one",
        )
        if merged["center"].isna().any():
            raise SystemExit(f"missing centers for {block.group}/d{block.horizon}: {int(merged['center'].isna().sum())}")
        idx = merged["index"].to_numpy(dtype="int64")
        center = merged["center"].to_numpy(dtype="float64") % 360.0
        hw = float(block.half_width)
        before_center = pd.to_numeric(merged["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        delta = circ_abs_diff(center, before_center)
        patch_key = f"{block.group}_d{block.horizon}_{block.candidate}"
        patch_counts[patch_key] = int(len(idx))
        stability.append(
            {
                "block": patch_key,
                "center_delta_mean": float(np.nanmean(delta)),
                "center_delta_p50": float(np.nanquantile(delta, 0.50)),
                "center_delta_p90": float(np.nanquantile(delta, 0.90)),
                "center_delta_p99": float(np.nanquantile(delta, 0.99)),
                "passed": bool(np.nanmean(delta) <= MAX_INFERENCE_CENTER_SHIFT_MEAN and np.nanquantile(delta, 0.90) <= MAX_INFERENCE_CENTER_SHIFT_P90),
            }
        )
        patched.loc[idx, "dir_50"] = center
        patched.loc[idx, "dir_05"] = (center - hw) % 360.0
        patched.loc[idx, "dir_95"] = (center + hw) % 360.0
    return patched, patch_counts, stability


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


def validate_delta(before: pd.DataFrame, after: pd.DataFrame, blocks: list[SelectedBlock]) -> dict[str, object]:
    speed_changed = rows_changed(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, after, DIR_COLS, 1, circular=True)
    allowed = np.zeros(len(after), dtype=bool)
    counts = {}
    for block in blocks:
        mask = (
            after["type"].eq("grid")
            & after["region"].eq(REGION)
            & after["horizon"].eq(block.horizon)
            & after["level"].isin(target_levels(block.group))
        ).to_numpy(dtype=bool)
        allowed |= mask
        counts[f"{block.group}_d{block.horizon}"] = int(mask.sum())
    outside = dir_changed & ~allowed
    if int(speed_changed.sum()) != 0:
        raise SystemExit(f"unexpected speed rows changed: {int(speed_changed.sum())}")
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected non-target direction rows changed: {int(outside.sum())}")
    return {
        "target_rows_by_block": counts,
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_direction_rows_changed": int(outside.sum()),
    }


def compliance_payload() -> dict[str, object]:
    return {
        "official_dataset_root": str(DATA),
        "external_training_data_used": False,
        "web_data_used": False,
        "evaluation_target_labels_used_for_training": False,
        "notes": [
            "CV uses only official training features/reanalysis with validation years held out chronologically.",
            "Surface analog labels come from official train target columns for historical origins only.",
            "Pressure analog labels come from official training pressure reanalysis for historical origins only.",
            "Final inference uses official inference feature parquets plus official historical train labels/reanalysis as the analog archive.",
            "No external datasets, hidden labels, or future target labels from the evaluation period are read.",
            "Public leaderboard metrics are used only as fixed gate thresholds and documentation, not as model features or labels.",
        ],
    }


def code_hashes() -> dict[str, str]:
    return {
        "build_ns_grid_dir_regime_fast_gate_candidate.py": sha256(Path(__file__).resolve()),
        "hres_mos_residual_branch.py": sha256(ROOT / "hres_mos_residual_branch.py"),
        "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
    }


def write_no_submission_manifest(summary: pd.DataFrame, reason: str) -> None:
    payload = {
        "status": "gate_failed_no_submission_written",
        "reason": reason,
        "cv_by_fold_csv": str(CV_BY_FOLD),
        "cv_summary_csv": str(CV_SUMMARY),
        "cv_summary": summary.to_dict(orient="records"),
        "compliance": compliance_payload(),
        "code_hashes": code_hashes(),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def write_submission_manifest(
    final: pd.DataFrame,
    summary: pd.DataFrame,
    blocks: list[SelectedBlock],
    stability: list[dict[str, object]],
    delta: dict[str, object],
    patch_counts: dict[str, object],
) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")
    payload = {
        "status": "submission_written_after_strict_cv_and_stability_gates",
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
        "accepted_blocks": [block.__dict__ for block in blocks],
        "patch_counts": patch_counts,
        "inference_stability": stability,
        "delta": delta,
        "cv_by_fold_csv": str(CV_BY_FOLD),
        "cv_summary_csv": str(CV_SUMMARY),
        "cv_summary": summary.to_dict(orient="records"),
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
        },
        "compliance": compliance_payload(),
        "code_hashes": code_hashes(),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def run_cv(
    feat: pd.DataFrame,
    means: pd.DataFrame,
    store: SurfaceTargetStore,
    cube: HM.CubeStore,
    cube_idx_map: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group, horizon in BLOCKS:
        for val_year in VAL_YEARS:
            print(f"CV {REGION}/{group}/d{horizon} val_year={val_year}", flush=True)
            rows.extend(evaluate_block(feat, means, store, cube, cube_idx_map, group, horizon, val_year))
    by_fold = pd.DataFrame(rows)
    summary = summarize_cv(by_fold)
    by_fold.to_csv(CV_BY_FOLD, index=False)
    summary.to_csv(CV_SUMMARY, index=False)
    print("Regime analog CV summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {CV_BY_FOLD}", flush=True)
    print(f"Wrote {CV_SUMMARY}", flush=True)
    return by_fold, summary


def main() -> None:
    require(BASE_CSV, "Run .\\run_ns_p7dir_mosres_e2e.ps1 first.")
    cleanup_outputs()

    feat = load_train_features()
    grid = build_grid(feat)
    feat = attach_grid_idx(feat, grid)
    means = feature_daily_means(feat)
    store = build_surface_store(feat, grid)
    cube, _ = load_pressure_cube()
    cube_idx_map = pressure_index_map(grid, cube)

    _, summary = run_cv(feat, means, store, cube, cube_idx_map)
    blocks = selected_blocks(summary)
    if not blocks:
        write_no_submission_manifest(summary, "No NS grid-direction analog block cleared the strict CV gates.")
        return

    print("Selected blocks:", [block.__dict__ for block in blocks], flush=True)
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    before = base.copy()
    patched, patch_counts, stability = apply_blocks(base, blocks, grid, means, store, cube, cube_idx_map)
    if any(not bool(item["passed"]) for item in stability):
        write_no_submission_manifest(
            summary,
            "One or more accepted CV blocks failed the final inference center-shift stability gate.",
        )
        return
    final = E2E.validate_final(patched)
    delta = validate_delta(before, final, blocks)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_submission_manifest(final, summary, blocks, stability, delta, patch_counts)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
