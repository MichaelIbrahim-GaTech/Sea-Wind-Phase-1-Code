"""Shared utilities for Phase 1 Wind Prediction starting kits.

This module provides functions used across all training notebooks:
- Feature loading and selection
- Vertical level expansion (10m → 7 levels)
- Prediction and submission generation
- Scoring metrics
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIONS = ["north_sea", "east_china_sea"]
HORIZONS = [1, 7, 14]
HOURS = [0, 6, 12, 18]
QUANTILES_DEFAULT = [0.05, 0.5, 0.95]
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
LEVEL_HEIGHTS = {1000: 100, 925: 800, 850: 1500, 700: 3000, 500: 5500}

# Default feature budget per horizon
TOP_K_DEFAULT = {1: 15, 7: 20, 14: 25}

# Worldwide feature prefixes (for excluding from speed models if desired)
WORLDWIDE_PREFIXES = (
    "nao_", "siberian_", "icelandic_", "ns_pressure_g",
    "ecs_pressure_g", "natl_pc", "wpac_pc", "up_",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_train_data(features_dir: Path | str, region: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load pre-computed training features for a region.

    Returns (DataFrame, feature_cols, speed_targets, dir_targets).
    """
    path = Path(features_dir) / f"train_{region}.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])

    target_cols = sorted([c for c in df.columns
                          if c.startswith(("speed_d", "dir_d"))])
    speed_targets = sorted([c for c in target_cols if c.startswith("speed_d")])
    dir_targets = sorted([c for c in target_cols if c.startswith("dir_d")])

    exclude = {"time"} | set(target_cols)
    feature_cols = sorted([
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float32, np.float64, np.int64, np.int32, float, int]
    ])

    return df, feature_cols, speed_targets, dir_targets


def load_inference_features(features_dir: Path | str, window_id: int, region: str) -> pd.DataFrame:
    """Load pre-computed inference features for one window/region."""
    path = Path(features_dir) / f"inference_window_{window_id}_{region}.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_vertical_ratios(features_dir: Path | str, region: str) -> Optional[pd.DataFrame]:
    """Load vertical wind profile ratios for level expansion."""
    path = Path(features_dir) / f"vertical_ratios_{region}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def exclude_worldwide_features(feature_cols: list[str]) -> list[str]:
    """Filter out worldwide features, keeping only local reanalysis + HRES."""
    return [c for c in feature_cols
            if not any(c.startswith(p) for p in WORLDWIDE_PREFIXES)]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def compute_feature_selection(df: pd.DataFrame, feature_cols: list[str], speed_targets: list[str],
                              model_type="lgbm", top_k=None,
                              subsample=200_000, random_state=42):
    """Compute per-target feature importance and select top-k features.

    Parameters
    ----------
    df : DataFrame with features and targets (training data)
    feature_cols : list of feature column names
    speed_targets : list of speed target column names
    model_type : 'lgbm', 'catboost', or 'qrf'
    top_k : dict {horizon: k} or None for defaults
    subsample : max rows for quick importance (None = all)

    Returns
    -------
    dict : {target_name: [selected_feature_names]}
    """
    if top_k is None:
        top_k = TOP_K_DEFAULT.copy()

    # Subsample for speed
    train_mask = df["time"].dt.year.isin([2019, 2020])
    sub = df[train_mask]
    if subsample and len(sub) > subsample:
        sub = sub.sample(n=subsample, random_state=random_state)

    selected = {}
    for tgt in speed_targets:
        horizon = int(tgt.split("_")[1][1:])
        k = top_k.get(horizon, 25)

        y = sub[tgt].dropna()
        X = sub.loc[y.index, feature_cols].fillna(0)

        if len(y) < 100:
            selected[tgt] = feature_cols[:k]
            continue

        if model_type == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(
                iterations=100, depth=5, learning_rate=0.1,
                random_seed=random_state, verbose=0,
            )
            m.fit(X, y)
            imp = pd.Series(m.feature_importances_, index=feature_cols)
        elif model_type == "qrf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(
                n_estimators=50, max_depth=10, min_samples_leaf=10,
                n_jobs=-1, random_state=random_state,
            )
            m.fit(X, y)
            imp = pd.Series(m.feature_importances_, index=feature_cols)
        else:  # lgbm (default)
            import lightgbm as lgb
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.5, colsample_bytree=0.8, min_child_samples=50,
                verbose=-1, n_jobs=-1,
            )
            m.fit(X, y)
            imp = pd.Series(m.feature_importances_, index=feature_cols)

        if k is None:  # use all features
            selected[tgt] = feature_cols
        else:
            selected[tgt] = imp.nlargest(k).index.tolist()

    return selected


def load_or_compute_selection(cache_path: Path | str, df: pd.DataFrame, feature_cols: list[str], speed_targets: list[str],
                              model_type="lgbm", top_k=None, force=False):
    """Load cached feature selection or compute and save it.

    Parameters
    ----------
    cache_path : Path to JSON cache file
    force : if True, recompute even if cache exists

    Returns
    -------
    dict : {target_name: [selected_feature_names]}
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            selected = json.load(f)
        print(f"Loaded cached feature selection: {cache_path.name} "
              f"({len(selected)} targets)")
        return selected

    print(f"Computing feature selection ({model_type})...")
    selected = compute_feature_selection(
        df, feature_cols, speed_targets,
        model_type=model_type, top_k=top_k,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved: {cache_path.name}")
    return selected


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------
def winkler_score(actual: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, alpha: float = 0.1) -> float:
    """Winkler interval score for a (1-alpha) prediction interval. Lower is better."""
    width = q_hi - q_lo
    below = actual < q_lo
    above = actual > q_hi
    penalty = np.where(below, (2 / alpha) * (q_lo - actual),
                       np.where(above, (2 / alpha) * (actual - q_hi), 0.0))
    return float(np.nanmean(width + penalty))


def circular_mae(actual_deg: np.ndarray, predicted_deg: np.ndarray) -> float:
    """Circular MAE handling 0/360 wrap-around."""
    diff = np.abs(actual_deg - predicted_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.nanmean(diff))


def compute_direction_intervals(train_df, dir_targets, feature_cols,
                                dir_models, quantile_lo=0.05, quantile_hi=0.95):
    """Compute empirical circular residual intervals for direction.

    Uses training residuals stratified by horizon to produce dir_05/dir_95
    offsets. At inference, dir_05 = dir_50 - offset, dir_95 = dir_50 + offset.

    Parameters
    ----------
    train_df : training DataFrame with direction targets
    dir_targets : list of direction target column names
    feature_cols : feature columns
    dir_models : {target: (m_sin, m_cos, feats)} direction models

    Returns
    -------
    dict : {horizon: (offset_lo, offset_hi)} in degrees
        dir_05 = (dir_50 - offset_hi) % 360
        dir_95 = (dir_50 + offset_hi) % 360
    """
    offsets = {}
    for tgt, (m_sin, m_cos, feats) in dir_models.items():
        horizon = int(tgt.split("_")[1][1:])

        y = train_df[tgt].dropna()
        X = train_df.loc[y.index, feats].fillna(0)

        # Predict direction
        pred_dir = np.degrees(np.arctan2(m_sin.predict(X), m_cos.predict(X))) % 360

        # Circular residuals (always positive, 0-180)
        diff = np.abs(y.values - pred_dir)
        residuals = np.minimum(diff, 360 - diff)

        # Use the quantile_hi percentile as the symmetric offset
        offset = float(np.percentile(residuals, quantile_hi * 100))
        offsets[tgt] = offset

    # Average offsets per horizon (across hours)
    horizon_offsets = {}
    for h in [1, 7, 14]:
        h_offsets = [v for k, v in offsets.items() if f"_d{h}_" in k]
        if h_offsets:
            horizon_offsets[h] = float(np.mean(h_offsets))

    return horizon_offsets


def add_direction_intervals(speed_dir_df, dir_offsets):
    """Add dir_05 and dir_95 columns to a predictions DataFrame.

    Parameters
    ----------
    speed_dir_df : DataFrame with dir_50, horizon columns
    dir_offsets : {horizon: offset_degrees} from compute_direction_intervals

    Returns
    -------
    DataFrame with dir_05 and dir_95 added
    """
    df = speed_dir_df.copy()
    df["dir_05"] = 0.0
    df["dir_95"] = 0.0

    for h, offset in dir_offsets.items():
        mask = df["horizon"] == h
        df.loc[mask, "dir_05"] = (df.loc[mask, "dir_50"] - offset) % 360
        df.loc[mask, "dir_95"] = (df.loc[mask, "dir_50"] + offset) % 360

    # Round
    df["dir_05"] = df["dir_05"].round(1)
    df["dir_95"] = df["dir_95"].round(1)
    return df


# ---------------------------------------------------------------------------
# Vertical level expansion
# ---------------------------------------------------------------------------
def power_law_ratio(level: int) -> float:
    """Fallback: power-law wind profile ratio = (height / 10m)^0.14."""
    h = LEVEL_HEIGHTS.get(level, 1000)
    return (h / 10.0) ** 0.14


def expand_to_all_levels(preds_10m, ratios_df, context_month):
    """Expand 10m predictions to all 7 vertical levels.

    Returns DataFrame with levels: 10m, 100m, 1000, 925, 850, 700, 500.
    Uses climatological vertical ratios where available, power-law fallback otherwise.
    """
    result = preds_10m.copy()
    result["level"] = "10m"
    level_frames = [result]

    # Helper to apply ratio for one level
    def _apply_ratio(preds, level_str, ratio_value_or_df):
        lev_df = preds.copy()
        lev_df["level"] = level_str

        if isinstance(ratio_value_or_df, pd.DataFrame) and len(ratio_value_or_df) > 0:
            r = ratio_value_or_df[["latitude", "longitude", "speed_ratio"]].copy()
            for c in ["latitude", "longitude"]:
                lev_df[c] = lev_df[c].astype(float).round(2)
                r[c] = r[c].astype(float).round(2)
            lev_df = lev_df.merge(r, on=["latitude", "longitude"], how="left")
            # Fallback for unmatched grid points
            if level_str == "100m":
                lev_df["speed_ratio"] = lev_df["speed_ratio"].fillna((100 / 10) ** 0.14)
            else:
                lev_df["speed_ratio"] = lev_df["speed_ratio"].fillna(
                    power_law_ratio(int(level_str)))
            # Direction climatology if available
            if "dir_clim" in ratio_value_or_df.columns:
                r_dir = ratio_value_or_df[["latitude", "longitude", "dir_clim"]].copy()
                for c in ["latitude", "longitude"]:
                    r_dir[c] = r_dir[c].round(2)
                lev_df = lev_df.merge(r_dir, on=["latitude", "longitude"], how="left",
                                       suffixes=("", "_clim"))
                if "dir_clim" in lev_df.columns:
                    has_clim = lev_df["dir_clim"].notna()
                    lev_df.loc[has_clim, "dir_50"] = lev_df.loc[has_clim, "dir_clim"].round(1)
                    lev_df.drop(columns=["dir_clim"], inplace=True, errors="ignore")
        else:
            if level_str == "100m":
                lev_df["speed_ratio"] = (100 / 10) ** 0.14
            else:
                lev_df["speed_ratio"] = power_law_ratio(int(level_str))

        for qcol in ["q05", "q50", "q95"]:
            lev_df[qcol] = (lev_df[qcol] * lev_df["speed_ratio"]).round(4)
        lev_df.drop(columns=["speed_ratio"], inplace=True, errors="ignore")
        return lev_df

    # 100m level
    if ratios_df is not None and "100m" in ratios_df["level"].values:
        r100 = ratios_df[(ratios_df["level"] == "100m") &
                         (ratios_df["month"] == context_month)]
        level_frames.append(_apply_ratio(preds_10m, "100m", r100))
    else:
        level_frames.append(_apply_ratio(preds_10m, "100m", None))

    # Pressure levels
    for lev in PRESSURE_LEVELS:
        if ratios_df is not None and str(lev) in ratios_df["level"].values:
            r = ratios_df[(ratios_df["level"] == str(lev)) &
                          (ratios_df["month"] == context_month)]
            level_frames.append(_apply_ratio(preds_10m, str(lev), r))
        else:
            level_frames.append(_apply_ratio(preds_10m, str(lev), None))

    return pd.concat(level_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_speed_10m(features_df, speed_models, selected_features,
                      feature_cols, quantiles=None):
    """Generate 10m speed predictions for all targets.

    Parameters
    ----------
    features_df : inference DataFrame (last day of context, 2565 rows)
    speed_models : {target: {quantile: model}} or {target: [seed_models]}
    selected_features : {target: [feature_list]}
    feature_cols : all feature columns (for ensuring existence)
    quantiles : list of quantile values used [lo, mid, hi]

    Returns
    -------
    DataFrame with columns: latitude, longitude, horizon, hour, q05, q50, q95
    """
    if quantiles is None:
        quantiles = QUANTILES_DEFAULT

    q_lo, q_mid, q_hi = quantiles[0], quantiles[1], quantiles[2]

    # Ensure all feature cols exist
    for c in feature_cols:
        if c not in features_df.columns:
            features_df[c] = 0.0

    rows = []
    for tgt, models in speed_models.items():
        horizon = int(tgt.split("_")[1][1:])
        hour = int(tgt.split("_")[2][1:])
        feats = selected_features.get(tgt, feature_cols)
        X = features_df[feats].fillna(0)

        # Handle multi-seed models (list of dicts) vs single model (dict)
        if isinstance(models, list):
            # Multi-seed: average across seeds
            q05 = np.mean([np.maximum(m[q_lo].predict(X), 0) for m in models], axis=0)
            q50 = np.mean([m[q_mid].predict(X) for m in models], axis=0)
            q95 = np.mean([m[q_hi].predict(X) for m in models], axis=0)
        else:
            q05 = np.maximum(models[q_lo].predict(X), 0)
            q50 = models[q_mid].predict(X)
            q95 = models[q_hi].predict(X)

        for j in range(len(features_df)):
            rows.append({
                "latitude": round(float(features_df.iloc[j]["latitude"]), 2),
                "longitude": round(float(features_df.iloc[j]["longitude"]), 2),
                "horizon": horizon, "hour": hour,
                "q05": round(float(q05[j]), 4),
                "q50": round(float(q50[j]), 4),
                "q95": round(float(q95[j]), 4),
            })

    return pd.DataFrame(rows)


def predict_direction(features_df, dir_models, feature_cols):
    """Generate direction predictions for all targets.

    Parameters
    ----------
    dir_models : {target: (model_sin, model_cos, dir_feature_list)}

    Returns
    -------
    dict : {(horizon, hour): array of direction predictions}
    """
    for c in feature_cols:
        if c not in features_df.columns:
            features_df[c] = 0.0

    dir_preds = {}
    for tgt, (m_sin, m_cos, dir_feats) in dir_models.items():
        horizon = int(tgt.split("_")[1][1:])
        hour = int(tgt.split("_")[2][1:])
        for c in dir_feats:
            if c not in features_df.columns:
                features_df[c] = 0.0
        X = features_df[dir_feats].fillna(0)
        dp = np.degrees(np.arctan2(m_sin.predict(X), m_cos.predict(X))) % 360
        dir_preds[(horizon, hour)] = dp

    return dir_preds


def merge_speed_direction(speed_df, dir_preds, default_dir=180.0):
    """Merge direction predictions onto speed predictions DataFrame."""
    speed_df = speed_df.copy()
    speed_df["dir_50"] = default_dir
    for (h, hr), dp in dir_preds.items():
        mask = (speed_df["horizon"] == h) & (speed_df["hour"] == hr)
        speed_df.loc[mask, "dir_50"] = np.round(dp, 1)
    return speed_df


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------
def attach_station_predictions(grid_sub, features_dir, station_metadata=None):
    """Convert a grid-only submission DataFrame into the unified format expected
    by the scorer (grid rows + station rows, with a leading ``type`` column).

    For each station, the baseline approach is to inherit the 10m prediction from
    its nearest grid point (via ``nearest_grid_lat/lon`` in ``station_metadata.csv``).
    Participants can override this by editing the station rows after this call.

    Parameters
    ----------
    grid_sub : DataFrame with columns: window, region, latitude, longitude,
        horizon, hour, level, q05, q50, q95, dir_50 (and optionally dir_05/dir_95).
        Values in ``level`` can be ``"10m"``, ``"100m"``, ``"1000"``, ``"925"``,
        ``"850"``, ``"700"``, ``"500"`` (strings or ints; will be coerced to str).
    features_dir : Path to features directory — used to auto-locate
        ``station_metadata.csv`` in the sibling ``scoring/`` folder.
    station_metadata : optional DataFrame override (same schema as the CSV).

    Returns
    -------
    DataFrame with an added ``type`` column (``"grid"`` or ``"station"``) and a
    ``station`` column (empty for grid rows). Column order matches what the
    scoring program expects.
    """
    grid_sub = grid_sub.copy()
    grid_sub["level"] = grid_sub["level"].astype(str)

    # Ensure direction intervals exist BEFORE building station rows — the Phase 1
    # scorer REQUIRES dir_05/dir_95 (circular Winkler on the 90% PI). If the
    # caller forgot to run add_direction_intervals(), fall back to a conservative
    # ±90° arc centred on dir_50 so the submission at least passes the schema
    # check. This must happen before the station merge so station rows inherit
    # the fallback, not an empty-string fill later on.
    if "dir_05" not in grid_sub.columns or "dir_95" not in grid_sub.columns:
        print("  WARNING: dir_05/dir_95 missing from grid submission — "
              "falling back to symmetric ±90° default. Run "
              "compute_direction_intervals() + add_direction_intervals() "
              "to produce calibrated intervals that will score better.")
        grid_sub["dir_05"] = (grid_sub["dir_50"] - 90.0) % 360
        grid_sub["dir_95"] = (grid_sub["dir_50"] + 90.0) % 360

    if station_metadata is None:
        station_metadata = _load_station_metadata(features_dir)

    # Grid side — tag with type + empty station
    grid_sub["type"] = "grid"
    grid_sub["station"] = ""

    # Station side — inherit 10m predictions from the nearest grid point per station
    station_rows = None
    if station_metadata is not None:
        grid_10m = grid_sub[grid_sub["level"] == "10m"].copy()
        grid_10m["latitude"] = grid_10m["latitude"].astype(float).round(2)
        grid_10m["longitude"] = grid_10m["longitude"].astype(float).round(2)

        merged = grid_10m.merge(
            station_metadata[["station", "region", "nearest_grid_lat", "nearest_grid_lon"]],
            left_on=["region", "latitude", "longitude"],
            right_on=["region", "nearest_grid_lat", "nearest_grid_lon"],
            how="inner",
            suffixes=("", "_meta"),
        )
        if len(merged) > 0:
            # Use the station column from metadata (overwrites the empty one from grid_sub)
            merged["station"] = merged["station_meta"] if "station_meta" in merged.columns else merged["station"]
            merged = merged.drop(columns=[c for c in ("station_meta", "nearest_grid_lat", "nearest_grid_lon") if c in merged.columns])
            merged["type"] = "station"
            merged["level"] = ""
            merged["latitude"] = np.nan
            merged["longitude"] = np.nan
            station_rows = merged

    # Unified output — dir_05/dir_95 are now required
    out_cols = ["type", "window", "region", "latitude", "longitude", "station",
                "horizon", "hour", "level", "q05", "q50", "q95",
                "dir_05", "dir_50", "dir_95"]

    for c in out_cols:
        if c not in grid_sub.columns:
            grid_sub[c] = np.nan if c in ("latitude", "longitude") else ""

    frames = [grid_sub[out_cols]]
    if station_rows is not None:
        for c in out_cols:
            if c not in station_rows.columns:
                station_rows[c] = np.nan if c in ("latitude", "longitude") else ""
        frames.append(station_rows[out_cols])

    return pd.concat(frames, ignore_index=True)


def _load_station_metadata(features_dir):
    """Auto-locate station_metadata.csv next to features/ (sibling scoring/ dir)."""
    features_dir = Path(features_dir)
    candidates = [
        features_dir.parent / "scoring" / "station_metadata.csv",
        features_dir / "station_metadata.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["nearest_grid_lat"] = df["nearest_grid_lat"].round(2)
            df["nearest_grid_lon"] = df["nearest_grid_lon"].round(2)
            return df
    return None


def _build_station_rows(preds_10m, station_meta, region, wid):
    """Map 10m grid predictions to station rows via nearest_grid_{lat,lon}.

    Baseline approach: each station inherits the prediction from its nearest
    grid point (the lat/lon the scoring program uses to locate the station).
    Participants are free to override this with station-specific calibration.
    """
    if station_meta is None:
        return None

    region_stations = station_meta[station_meta["region"] == region]
    if len(region_stations) == 0:
        return None

    p = preds_10m.copy()
    p["latitude"] = p["latitude"].astype(float).round(2)
    p["longitude"] = p["longitude"].astype(float).round(2)

    station_rows = p.merge(
        region_stations[["station", "nearest_grid_lat", "nearest_grid_lon"]],
        left_on=["latitude", "longitude"],
        right_on=["nearest_grid_lat", "nearest_grid_lon"],
        how="inner",
    )
    if len(station_rows) == 0:
        return None

    station_rows["window"] = wid
    station_rows["region"] = region
    station_rows["type"] = "station"
    station_rows["level"] = ""
    # Null out grid-specific columns so the unified CSV is unambiguous
    station_rows["latitude"] = np.nan
    station_rows["longitude"] = np.nan
    station_rows = station_rows.drop(columns=["nearest_grid_lat", "nearest_grid_lon"])
    return station_rows


def generate_submission(speed_models: dict, dir_models: dict, selected_features: dict,
                        feature_cols_all, vertical_ratios,
                        features_dir, regions=None, n_windows=8,
                        quantiles=None, blend_models=None, blend_weights=None,
                        dir_offsets=None, include_stations=True,
                        station_metadata=None):
    """Generate a complete unified-format submission.

    The output contains two kinds of rows distinguished by the ``type`` column:

    - ``type == "grid"``: one row per (window, region, lat, lon, horizon, hour, level)
      across the 7 vertical levels (10m, 100m, 1000, 925, 850, 700, 500).
    - ``type == "station"``: one row per (window, region, station, horizon, hour).
      Baseline: each station inherits the 10m prediction at its nearest grid point.

    Parameters
    ----------
    speed_models : {region: {target: models}}
    dir_models : {region: {target: (m_sin, m_cos, feats)}}
    selected_features : {region: {target: [feats]}}
    feature_cols_all : {region: [all_feature_cols]}
    vertical_ratios : {region: DataFrame or None}
    features_dir : Path to features directory
    blend_models : optional {region: {target: models}} for blending
    blend_weights : optional {region: float} CatBoost weight (1-w for blend)
    dir_offsets : optional {region: {horizon: offset_deg}} for direction intervals
    include_stations : if True (default), emit station rows alongside grid rows
    station_metadata : optional DataFrame with columns [station, region,
        nearest_grid_lat, nearest_grid_lon]. Auto-loaded from ``features_dir``'s
        sibling ``scoring/station_metadata.csv`` if not provided.
    """
    if regions is None:
        regions = REGIONS
    if quantiles is None:
        quantiles = QUANTILES_DEFAULT

    if include_stations and station_metadata is None:
        station_metadata = _load_station_metadata(features_dir)
        if station_metadata is None:
            print("  WARNING: station_metadata.csv not found — station rows will be skipped")

    grid_frames = []
    station_frames = []

    for wid in range(1, n_windows + 1):
        for region in regions:
            df_inf = load_inference_features(features_dir, wid, region)
            context_month = int(df_inf["time"].max().month)
            fcols = feature_cols_all[region]

            preds_10m = predict_speed_10m(
                df_inf, speed_models[region], selected_features[region],
                fcols, quantiles=quantiles,
            )

            if blend_models and blend_weights and region in blend_models:
                w = blend_weights[region]
                preds_blend = predict_speed_10m(
                    df_inf, blend_models[region], selected_features[region],
                    fcols, quantiles=quantiles,
                )
                for qcol in ["q05", "q50", "q95"]:
                    preds_10m[qcol] = (w * preds_10m[qcol] +
                                       (1 - w) * preds_blend[qcol])

            stacked = preds_10m[["q05", "q50", "q95"]].to_numpy(copy=True)
            stacked.sort(axis=1)
            preds_10m["q05"] = stacked[:, 0]
            preds_10m["q50"] = stacked[:, 1]
            preds_10m["q95"] = stacked[:, 2]

            dir_preds = predict_direction(df_inf, dir_models[region], fcols)
            preds_10m = merge_speed_direction(preds_10m, dir_preds)

            # Direction intervals are REQUIRED by the Phase 1 scorer
            # (circular Winkler on the 90% PI). If the caller provided
            # offsets, use them; otherwise fall back to a conservative ±90°
            # default so the submission at least validates.
            if dir_offsets and region in dir_offsets:
                preds_10m = add_direction_intervals(preds_10m, dir_offsets[region])
            else:
                fallback_offsets = {h: 90.0 for h in (1, 7, 14)}
                preds_10m = add_direction_intervals(preds_10m, fallback_offsets)

            # Station rows — built from the 10m grid before vertical expansion
            if include_stations and station_metadata is not None:
                station_block = _build_station_rows(preds_10m, station_metadata, region, wid)
                if station_block is not None:
                    station_frames.append(station_block)

            # Grid rows — expand 10m to 7 vertical levels
            preds = expand_to_all_levels(
                preds_10m, vertical_ratios.get(region), context_month,
            )
            preds["window"] = wid
            preds["region"] = region
            preds["type"] = "grid"
            preds["station"] = ""
            grid_frames.append(preds)

            n_levels = preds["level"].nunique()
            n_station = len(station_block) if include_stations and station_metadata is not None and station_block is not None else 0
            print(f"  W{wid} {region}: grid={len(preds):,} ({n_levels} levels), station={n_station}")

    grid_sub = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    station_sub = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame()

    # Unified column order — dir_05/dir_95 are REQUIRED by the scorer
    # (circular Winkler on the 90% PI).
    out_cols = ["type", "window", "region", "latitude", "longitude", "station",
                "horizon", "hour", "level", "q05", "q50", "q95",
                "dir_05", "dir_50", "dir_95"]

    for df in (grid_sub, station_sub):
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan if c in ("latitude", "longitude") else ""

    sub = pd.concat(
        [grid_sub[out_cols], station_sub[out_cols]] if not station_sub.empty else [grid_sub[out_cols]],
        ignore_index=True,
    )

    # Force level to string so the CSV round-trips cleanly (pandas would otherwise
    # auto-infer mixed int/str on read for pressure levels that look numeric).
    sub["level"] = sub["level"].astype(str)

    # Enforce constraints (applies uniformly to grid + station rows)
    sub["q05"] = sub["q05"].clip(lower=0)
    sub["q95"] = sub[["q50", "q95"]].max(axis=1)
    sub["q05"] = sub[["q05", "q50"]].min(axis=1)
    sub["dir_05"] = sub["dir_05"] % 360
    sub["dir_50"] = sub["dir_50"] % 360
    sub["dir_95"] = sub["dir_95"] % 360

    return sub
