"""Feature engineering functions for Phase 1 Wind Prediction.

This module contains all feature computation functions used by
1_feature_engineering.ipynb. Extracted to avoid notebook bloat
and enable reuse across notebooks.

Functions:
- compute_wind_speed_direction: u/v → speed/direction
- add_lag_features: 1/3/7 day lags
- add_rolling_features: 3/7 day rolling mean/std
- add_temporal_features: week-of-year cyclical encoding
- add_elevation: elevation data merge
- add_hres_features: HRES NWP forecast merge + has_hres indicator
- pivot_subdaily_features: 6-hourly → daily base with h6/h18 columns
- drop_redundant_features: remove correlated/constant features
- build_features: full pipeline (raw → model-ready)
- load_worldwide_data: load worldwide reanalysis parquets
- compute_teleconnection_proxies: NAO, Siberian High, pressure gradients
- compute_mslp_pca: PCA of hemispheric MSLP fields
- compute_upstream_features: MSLP/wind at upstream anchor points
- build_worldwide_features: orchestrate all worldwide feature computation
- merge_worldwide_features: merge worldwide features onto regional data
- build_inference_features: feature engineering for inference windows
- compute_vertical_ratios: monthly ws(level)/ws(10m) ratios
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.decomposition import PCA

# Constants
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
LEVEL_HEIGHTS = {1000: 100, 925: 800, 850: 1500, 700: 3000, 500: 5500}


def compute_wind_speed_direction(df):
    """Compute wind speed and meteorological direction from u/v components.

    Meteorological convention: direction wind comes FROM, clockwise from North.
    0=N, 90=E, 180=S, 270=W.
    """
    df = df.copy()
    # 10m wind
    df["ws10"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df["wd10"] = (270 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360
    # 100m wind
    if "u100" in df.columns and "v100" in df.columns:
        df["ws100"] = np.sqrt(df["u100"]**2 + df["v100"]**2)
        df["wd100"] = (270 - np.degrees(np.arctan2(df["v100"], df["u100"]))) % 360
    # Wind shear (100m vs 10m)
    if "ws100" in df.columns:
        df["wind_shear"] = df["ws100"] - df["ws10"]
    return df


def add_lag_features(df, group_cols=("latitude", "longitude"), lags_days=[0, 1, 3, 7]):
    """Add lag features per grid point. Lags are in DAYS (data is daily).

    lag=0  -> current day (explicit feature for the model)
    lag=1  -> 1 day ago
    lag=3  -> 3 days ago
    lag=7  -> 7 days ago
    """
    df = df.sort_values([*group_cols, "time"]).copy()
    lag_vars = ["ws10", "wd10", "msl", "t2m", "sshf", "z700", "z850"]
    for var in lag_vars:
        if var not in df.columns:
            continue
        for lag in lags_days:
            col_name = f"{var}_lag{lag}d"
            if lag == 0:
                df[col_name] = df[var]
            else:
                df[col_name] = df.groupby(list(group_cols))[var].shift(lag)
    return df




def add_rolling_features(df, group_cols=("latitude", "longitude"), windows_days=[3, 7]):
    """Add rolling mean and std of wind speed. Windows in DAYS.

    window=3  -> 3-day rolling
    window=7  -> 7-day rolling
    """
    df = df.sort_values([*group_cols, "time"]).copy()
    for w in windows_days:
        grp = df.groupby(list(group_cols))["ws10"]
        df[f"ws10_rmean{w}d"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"ws10_rstd{w}d"]  = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
    return df




def add_temporal_features(df):
    """Add temporal encodings (cyclical).

    Uses week-of-year (52 values) instead of day-of-year (365 values)
    to reduce overfitting risk while preserving seasonal signal.
    No raw 'month' integer — woy_sin/cos already captures seasonality
    with proper cyclical encoding (Dec→Jan wrap handled).
    """
    df = df.copy()
    woy = df["time"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.1775)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.1775)
    return df


def add_elevation(df, region, train_dir):
    """Merge elevation onto reanalysis grid points."""
    import xarray as xr
    elev_path = train_dir / f"elevation_{region}.nc"
    if not elev_path.exists():
        print(f"  elevation data not found at {elev_path}, skipping elevation")
        df["elevation"] = 0.0
        return df
    ds = xr.open_dataset(elev_path)
    z = ds["z"] if "z" in ds else ds[list(ds.data_vars)[0]]
    lat_dim = "lat" if "lat" in z.dims else "latitude"
    lon_dim = "lon" if "lon" in z.dims else "longitude"
    elev_df = z.to_dataframe().reset_index().rename(
        columns={lat_dim: "latitude", lon_dim: "longitude", z.name: "elevation"}
    )
    elev_df["latitude"] = elev_df["latitude"].round(2)
    elev_df["longitude"] = elev_df["longitude"].round(2)
    elev_df = elev_df.groupby(["latitude", "longitude"])["elevation"].mean().reset_index()
    ds.close()
    df = df.merge(elev_df, on=["latitude", "longitude"], how="left")
    df["elevation"] = df["elevation"].fillna(0.0)
    return df




def add_hres_features(df, region, data_dir):
    """Merge HRES NWP forecast columns (fcst_speed_d*_h*, fcst_dir_d*_h*).

    HRES provides forecasts at +1d and +7d (not +14d, which is beyond its range).
    Adds a `has_hres` binary indicator so models can distinguish "no forecast
    available" from "forecast = 0 m/s".
    """
    hres_path = data_dir / f"hres_{region}.parquet"
    if not hres_path.exists():
        print(f"  HRES not found at {hres_path}, skipping NWP features")
        df["has_hres"] = 0
        return df
    hres = pd.read_parquet(hres_path)
    hres["time"] = pd.to_datetime(hres["time"])
    for c in ["latitude", "longitude"]:
        hres[c] = hres[c].round(2)
        df[c] = df[c].round(2)
    merge_cols = ["time", "latitude", "longitude"]
    fcst_cols = [c for c in hres.columns if c.startswith("fcst_")]
    hres_subset = hres[merge_cols + fcst_cols].copy()
    hres_subset["has_hres"] = 1
    df = df.merge(hres_subset, on=merge_cols, how="left")
    df["has_hres"] = df["has_hres"].fillna(0).astype(int)
    n_matched = (df["has_hres"] == 1).sum()
    print(f"  HRES: merged {len(fcst_cols)} forecast columns, "
          f"{n_matched:,}/{len(df):,} rows matched ({n_matched/len(df)*100:.0f}%)")
    return df


def add_hres_pressure_features(df, region, data_dir):
    """Merge HRES pressure-level forecast columns (fcst_u_{lev}_d*_h*, fcst_v_{lev}_d*_h*).

    Pressure-level HRES provides u/v wind components at 5 pressure levels
    (1000, 925, 850, 700, 500 hPa) for horizons d1/d7/d10 (not d14). The
    ``time`` column is the 00Z init time, the same convention as the surface
    HRES file, so the merge on ``(time, latitude, longitude)`` produces
    forecasts whose lead times match the targets already in ``df``.

    Adds a ``has_hres_pressure`` binary indicator.
    """
    hres_path = data_dir / f"hres_pressure_{region}.parquet"
    if not hres_path.exists():
        print(f"  HRES pressure not found at {hres_path}, skipping pressure NWP features")
        df["has_hres_pressure"] = 0
        return df
    hres = pd.read_parquet(hres_path)
    hres["time"] = pd.to_datetime(hres["time"])
    for c in ["latitude", "longitude"]:
        hres[c] = hres[c].round(2)
        df[c] = df[c].round(2)
    merge_cols = ["time", "latitude", "longitude"]
    fcst_cols = [c for c in hres.columns if c.startswith(("fcst_u_", "fcst_v_"))]
    hres_subset = hres[merge_cols + fcst_cols].copy()
    hres_subset["has_hres_pressure"] = 1
    df = df.merge(hres_subset, on=merge_cols, how="left")
    df["has_hres_pressure"] = df["has_hres_pressure"].fillna(0).astype(int)
    n_matched = (df["has_hres_pressure"] == 1).sum()
    print(f"  HRES pressure: merged {len(fcst_cols)} forecast columns, "
          f"{n_matched:,}/{len(df):,} rows matched ({n_matched/len(df)*100:.0f}%)")
    return df



def pivot_subdaily_features(df, subdaily_vars=None):
    """Pivot 6-hourly data into daily base rows (00Z) with sub-daily columns.

    For each variable in subdaily_vars, adds columns like ws10_h6, ws10_h12, ws10_h18
    from the same day, merged onto the 00Z row. Also adds daily min/max/range for ws10.

    Returns only the 00Z rows with extra sub-daily columns.
    """
    if subdaily_vars is None:
        subdaily_vars = ["ws10", "wd10", "msl", "t2m"]

    df = df.copy()
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour

    # Base: 00Z rows
    base = df[df["hour"] == 0].drop(columns=["hour"]).copy()

    # Pivot sub-daily hours
    for hour in [6, 12, 18]:
        sub = df[df["hour"] == hour][["date", "latitude", "longitude"] + subdaily_vars].copy()
        sub = sub.rename(columns={v: f"{v}_h{hour}" for v in subdaily_vars})
        base = base.merge(sub, on=["date", "latitude", "longitude"], how="left")

    # Daily aggregates for wind speed (captures intra-day variability)
    daily_agg = df.groupby(["date", "latitude", "longitude"]).agg(
        ws10_daily_min=("ws10", "min"),
        ws10_daily_max=("ws10", "max"),
        ws10_daily_mean=("ws10", "mean"),
    ).reset_index()
    daily_agg["ws10_daily_range"] = daily_agg["ws10_daily_max"] - daily_agg["ws10_daily_min"]
    base = base.merge(daily_agg, on=["date", "latitude", "longitude"], how="left")

    base.drop(columns=["date"], inplace=True)
    return base




def build_features(reanalysis_df, region, train_dir):
    """Full feature engineering pipeline: raw reanalysis 6h -> daily model-ready features."""
    df = compute_wind_speed_direction(reanalysis_df)
    df = add_temporal_features(df)

    # Pivot to daily base rows (00Z) with sub-daily features
    df = pivot_subdaily_features(df, subdaily_vars=["ws10", "wd10", "msl", "t2m"])
    print(f"  After daily pivot: {len(df):,} rows (was {len(reanalysis_df):,})")

    # Lags and rolling stats operate on daily data now
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_elevation(df, region, train_dir)
    df = add_hres_features(df, region, train_dir)
    df = add_hres_pressure_features(df, region, train_dir)

    # Drop redundant features
    df = drop_redundant_features(df)
    return df




def drop_redundant_features(df):
    """Drop features that are redundant or risk overfitting.

    Removes ~20 columns:
    - lag0d duplicates (ws10_lag0d == ws10, etc.)
    - z850 block (z850 ≈ z700 with r=0.999)
    - ws10_daily_min (≈ ws10_daily_mean, keep max+mean+range)
    - Sub-daily h12 (keep h6+h18 for max spread)
    - has_hres (constant=1 after daily pivot, zero information)
    - Raw u/v components (redundant with ws/wd derived features)
    """
    drop_cols = []

    # 1. lag0d duplicates (raw column == lag0d)
    for var in ["ws10", "wd10", "msl", "t2m", "sshf", "z700", "z850"]:
        lag0 = f"{var}_lag0d"
        if lag0 in df.columns:
            drop_cols.append(lag0)

    # 2. z850 block — highly correlated with z700
    for suffix in ["", "_lag1d", "_lag3d", "_lag7d"]:
        col = f"z850{suffix}"
        if col in df.columns:
            drop_cols.append(col)

    # 3. Consolidate sub-daily: drop h12 (keep h6 + h18 for max spread)
    for var in ["ws10", "wd10", "t2m", "msl"]:
        col = f"{var}_h12"
        if col in df.columns:
            drop_cols.append(col)

    # 4. ws10_daily_min (keep max, mean, range)
    if "ws10_daily_min" in df.columns:
        drop_cols.append("ws10_daily_min")

    # 5. has_hres / has_hres_pressure — constant (all 1) after daily pivot, zero information
    for col in ("has_hres", "has_hres_pressure"):
        if col in df.columns:
            drop_cols.append(col)

    # 6. Raw u/v components — redundant with derived ws/wd/wind_shear
    for var in ["u10", "v10", "u100", "v100"]:
        if var in df.columns:
            drop_cols.append(var)

    # Only drop columns that exist, deduplicate
    drop_cols = sorted(set(c for c in drop_cols if c in df.columns))

    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} redundant features: {drop_cols}")

    return df




def load_stations_train(train_dir, region=None):
    """Load the training station observations (6-hourly, 2019-2021).

    Parameters
    ----------
    train_dir : Path
        The dataset's ``train/`` directory.
    region : str, optional
        If given (``"north_sea"`` or ``"east_china_sea"``), return only that
        region. Otherwise concatenate both.

    Returns
    -------
    DataFrame with columns:
        time, station, region, latitude, longitude, height_m, speed, direction

    Notes
    -----
    - NS_01 (FINO1) reports wind speed only; its ``direction`` column is
      always NaN. All other stations have near-complete direction coverage
      after the 6-hour aggregation.
    - Several BSH stations (NS_02, NS_07, NS_08) have sparse raw feeds;
      their row counts are well below the theoretical 4,384 bins per
      3 years. This is a genuine sensor-downtime artifact.
    """
    train_dir = Path(train_dir)
    regions = [region] if region else ("north_sea", "east_china_sea")
    frames = []
    for r in regions:
        path = train_dir / f"stations_{r}_6h.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found")
            continue
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
        print(f"  Loaded {path.name}: {len(df):,} rows "
              f"({df['station'].nunique()} stations)")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_stations_context(window_dir, region=None):
    """Load per-window station observations for one inference window.

    Returns rows from ``inference/window_{N}/context_stations_{region}.parquet``
    which contain station observations in
    ``[context_start - 30 days, context_end]``. By construction these files
    never contain data past the window's ``context_end``, so there is no
    future-of-inference leakage even if you use all rows for feature
    engineering.

    Parameters
    ----------
    window_dir : Path
        Path to one ``inference/window_{N}/`` directory.
    region : str, optional
        If given, load only that region's file; otherwise concatenate both.

    Returns
    -------
    DataFrame or None
    """
    window_dir = Path(window_dir)
    regions = [region] if region else ("north_sea", "east_china_sea")
    frames = []
    for r in regions:
        path = window_dir / f"context_stations_{r}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_worldwide_context(window_dir):
    """Load the per-window worldwide daily context file.

    Returns the DataFrame from
    ``{window_dir}/context_worldwide_daily.parquet`` or None if missing.
    The file contains worldwide daily reanalysis data from roughly 30 days
    before the window's ``context_start`` through its ``context_end``
    (inclusive) — no dates from the window's prediction period are included.
    """
    path = window_dir / "context_worldwide_daily.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  Loaded {path.name}: {len(df):,} rows "
          f"({df['time'].min().date()} → {df['time'].max().date()})")
    return df


def load_hres_pressure_train(train_dir, region):
    """Load the training HRES pressure-level forecast file for one region.

    Returns the DataFrame from ``{train_dir}/hres_pressure_{region}.parquet``
    (covers 2019-2021) or None if missing. Columns are
    ``fcst_u_{lev}_d{horizon}_h{hour}`` / ``fcst_v_{lev}_d{horizon}_h{hour}`` for
    levels ``1000 / 925 / 850 / 700 / 500 hPa`` and horizons d1 / d7 / d10.
    """
    path = Path(train_dir) / f"hres_pressure_{region}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  Loaded {path.name}: {len(df):,} rows "
          f"({df['time'].min().date()} → {df['time'].max().date()})")
    return df


def load_hres_pressure_context(window_dir, region):
    """Load the per-window HRES pressure-level context file for one region.

    Returns the DataFrame from
    ``{window_dir}/context_hres_pressure_{region}.parquet`` or None if missing.
    Same schema as :func:`load_hres_pressure_train`.
    """
    path = window_dir / f"context_hres_pressure_{region}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  Loaded {path.name}: {len(df):,} rows "
          f"({df['time'].min().date()} → {df['time'].max().date()})")
    return df


def load_worldwide_data(train_dir, years=None):
    """Load and concatenate worldwide reanalysis daily parquets from train/.

    The 2022 evaluation year is intentionally NOT shipped as a full-year file
    (it would leak future-of-inference dates). For inference, use
    ``load_worldwide_context(window_dir)`` to load the per-window worldwide
    context file instead, then concatenate with the training years.
    """
    if years is None:
        years = [2019, 2020, 2021]
    dfs = []
    for year in years:
        path = train_dir / f"reanalysis_worldwide_daily_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["time"] = pd.to_datetime(df["time"])
            dfs.append(df)
            print(f"  Loaded {path.name}: {len(df):,} rows")
        else:
            print(f"  WARNING: {path.name} not found")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)




def compute_teleconnection_proxies(ww):
    """Compute daily teleconnection index proxies from worldwide MSLP.

    Returns a DataFrame with columns: time, nao_proxy, siberian_high, icelandic_low,
    azores_high, ns_pressure_gradient, ecs_pressure_gradient.
    """
    ww = ww.copy()
    ww["latitude"] = ww["latitude"].round(0).astype(int)
    ww["longitude"] = ww["longitude"].round(0).astype(int)

    def extract_point(lat, lon):
        mask = (ww["latitude"] == lat) & (ww["longitude"] == lon)
        return ww[mask][["time", "msl"]].set_index("time")["msl"]

    # NAO proxy: standardized(Azores MSLP) - standardized(Iceland MSLP)
    msl_iceland = extract_point(65, -23)
    msl_azores = extract_point(38, -26)

    # Siberian High (driver of East Asian Winter Monsoon)
    msl_siberia = extract_point(50, 90)

    # North Sea pressure gradient (France - Norway)
    msl_norway = extract_point(65, 0)
    msl_france = extract_point(45, 0)

    # East China Sea pressure gradient (Siberia - subtropical)
    msl_subtropical = extract_point(25, 125)
    msl_high_lat = extract_point(50, 125)

    # Build DataFrame
    idx = pd.DataFrame({"time": sorted(ww["time"].unique())}).set_index("time")

    for name, series in [("msl_iceland", msl_iceland), ("msl_azores", msl_azores),
                         ("msl_siberia", msl_siberia), ("msl_norway", msl_norway),
                         ("msl_france", msl_france), ("msl_subtropical", msl_subtropical),
                         ("msl_high_lat", msl_high_lat)]:
        idx = idx.join(series.rename(name), how="left")

    # Standardize each
    for col in [c for c in idx.columns if c.startswith("msl_")]:
        mean, std = idx[col].mean(), idx[col].std()
        idx[f"{col}_std"] = (idx[col] - mean) / std if std > 0 else 0

    # NAO proxy
    idx["nao_proxy"] = idx["msl_azores_std"] - idx["msl_iceland_std"]

    # Siberian High index (standardized)
    idx["siberian_high"] = idx["msl_siberia_std"]

    # Icelandic Low (inverted — lower = stronger)
    idx["icelandic_low"] = -idx["msl_iceland_std"]

    # Large-scale pressure gradients
    idx["ns_pressure_gradient"] = (idx["msl_france"] - idx["msl_norway"]) / 20.0
    idx["ecs_pressure_gradient"] = (idx["msl_subtropical"] - idx["msl_high_lat"]) / 25.0

    result = idx[["nao_proxy", "siberian_high", "icelandic_low",
                  "ns_pressure_gradient", "ecs_pressure_gradient"]].reset_index()
    print(f"  Teleconnection proxies: {len(result)} days, "
          f"NAO range [{result['nao_proxy'].min():.2f}, {result['nao_proxy'].max():.2f}]")
    return result




def compute_mslp_pca(ww, domain_name, lat_range, lon_range, n_components=6):
    """Compute PCA of daily MSLP anomalies over a domain.

    Returns DataFrame with columns: time, {domain}_pc1..pc{n}, plus the fitted PCA object.
    """
    ww_lat = ww["latitude"].round(0).astype(int)
    ww_lon = ww["longitude"].round(0).astype(int)

    mask = ((ww_lat >= lat_range[0]) & (ww_lat <= lat_range[1]) &
            (ww_lon >= lon_range[0]) & (ww_lon <= lon_range[1]))
    domain = ww[mask].copy()
    domain["latitude"] = ww_lat[mask]
    domain["longitude"] = ww_lon[mask]

    # Pivot to (n_days, n_gridpoints)
    pivot = domain.pivot_table(index="time", columns=["latitude", "longitude"], values="msl")
    pivot = pivot.dropna(axis=1)  # drop any columns with NaN

    if pivot.shape[1] < n_components:
        print(f"  WARNING: {domain_name} has only {pivot.shape[1]} grid points, skipping PCA")
        return None, None

    # Area-weight by cos(latitude)
    lats = np.array([c[0] for c in pivot.columns])
    weights = np.sqrt(np.cos(np.radians(lats)))

    X = pivot.values * weights[np.newaxis, :]
    X_anom = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(X_anom)

    pc_df = pd.DataFrame(
        pc_scores,
        columns=[f"{domain_name}_pc{i+1}" for i in range(n_components)],
        index=pivot.index
    ).reset_index()

    var_explained = pca.explained_variance_ratio_[:n_components].sum() * 100
    print(f"  {domain_name} PCA: {n_components} components explain "
          f"{var_explained:.1f}% variance ({pivot.shape[1]} grid points)")
    return pc_df, pca




def compute_upstream_features(ww, region):
    """Extract MSLP and wind at upstream anchor points with lags.

    Weather systems propagate roughly west→east, so upstream features
    provide advance warning of incoming weather patterns.
    """
    UPSTREAM_POINTS = {
        "north_sea": {
            "iceland":      (65, -20),
            "mid_atlantic":  (55, -30),
            "greenland_tip": (60, -45),
            "iberian":       (40, -10),
            "north_atlantic": (50, -15),
        },
        "east_china_sea": {
            "central_china": (30, 105),
            "siberia":       (50, 90),
            "mongolia":      (45, 110),
            "japan_sea":     (40, 135),
            "south_china":   (20, 115),
        },
    }

    points = UPSTREAM_POINTS.get(region, {})
    if not points:
        return None

    ww = ww.copy()
    ww["latitude"] = ww["latitude"].round(0).astype(int)
    ww["longitude"] = ww["longitude"].round(0).astype(int)

    result = pd.DataFrame({"time": sorted(ww["time"].unique())})

    for name, (lat, lon) in points.items():
        mask = (ww["latitude"] == lat) & (ww["longitude"] == lon)
        point = ww[mask][["time", "msl", "u10", "v10"]].copy()
        point = point.rename(columns={
            "msl": f"up_{name}_msl",
            "u10": f"up_{name}_u10",
            "v10": f"up_{name}_v10",
        })
        # Compute wind speed at upstream point
        point[f"up_{name}_ws10"] = np.sqrt(point[f"up_{name}_u10"]**2 + point[f"up_{name}_v10"]**2)
        point = point.drop(columns=[f"up_{name}_u10", f"up_{name}_v10"])
        result = result.merge(point[["time", f"up_{name}_msl", f"up_{name}_ws10"]],
                              on="time", how="left")

    # Add lagged versions (1, 3, 5 day lags)
    upstream_cols = [c for c in result.columns if c.startswith("up_")]
    result = result.sort_values("time")
    for col in upstream_cols:
        for lag in [1, 3, 5]:
            result[f"{col}_lag{lag}d"] = result[col].shift(lag)

    n_features = len([c for c in result.columns if c != "time"])
    print(f"  {region} upstream features: {n_features} columns "
          f"({len(points)} points × 2 vars × 4 lags)")
    return result




def build_worldwide_features(train_dir, years=None):
    """Compute all worldwide features: teleconnections, MSLP PCA, upstream.

    Returns a dict with keys: teleconnections (DataFrame), pca_north_atlantic (DataFrame),
    pca_west_pacific (DataFrame), upstream_north_sea (DataFrame), upstream_ecs (DataFrame).
    """
    print("Loading worldwide reanalysis data...")
    ww = load_worldwide_data(train_dir, years)
    if ww is None:
        print("  No worldwide data found, skipping worldwide features")
        return {}

    result = {}

    # 1. Teleconnection proxies
    print("\nComputing teleconnection proxies...")
    result["teleconnections"] = compute_teleconnection_proxies(ww)

    # 2. MSLP PCA over two domains
    print("\nComputing MSLP PCA...")
    # North Atlantic domain (for North Sea predictions)
    pca_na, _ = compute_mslp_pca(ww, "natl", lat_range=(20, 80), lon_range=(-80, 40), n_components=6)
    result["pca_north_atlantic"] = pca_na

    # West Pacific domain (for East China Sea predictions)
    pca_wp, _ = compute_mslp_pca(ww, "wpac", lat_range=(10, 60), lon_range=(90, 180), n_components=6)
    result["pca_west_pacific"] = pca_wp

    # 3. Upstream features
    print("\nComputing upstream features...")
    result["upstream_north_sea"] = compute_upstream_features(ww, "north_sea")
    result["upstream_ecs"] = compute_upstream_features(ww, "east_china_sea")

    del ww
    return result




def merge_worldwide_features(df, worldwide_feats, region):
    """Merge worldwide features onto regional training/inference data.

    Selects region-appropriate features (North Atlantic PCA for NS, West Pacific for ECS).
    """
    if not worldwide_feats:
        return df

    df = df.copy()
    # Normalize time to date for merging (worldwide is daily)
    df["_merge_date"] = df["time"].dt.normalize()

    # Teleconnections (same for both regions, but different gradients are relevant)
    if "teleconnections" in worldwide_feats:
        tc = worldwide_feats["teleconnections"].copy()
        tc["_merge_date"] = pd.to_datetime(tc["time"]).dt.normalize()
        tc = tc.drop(columns=["time"])
        df = df.merge(tc, on="_merge_date", how="left")

    # Region-specific PCA
    if region == "north_sea" and "pca_north_atlantic" in worldwide_feats:
        pca = worldwide_feats["pca_north_atlantic"].copy()
        pca["_merge_date"] = pd.to_datetime(pca["time"]).dt.normalize()
        pca = pca.drop(columns=["time"])
        df = df.merge(pca, on="_merge_date", how="left")
    elif region == "east_china_sea" and "pca_west_pacific" in worldwide_feats:
        pca = worldwide_feats["pca_west_pacific"].copy()
        pca["_merge_date"] = pd.to_datetime(pca["time"]).dt.normalize()
        pca = pca.drop(columns=["time"])
        df = df.merge(pca, on="_merge_date", how="left")

    # Region-specific upstream
    upstream_key = "upstream_north_sea" if region == "north_sea" else "upstream_ecs"
    if upstream_key in worldwide_feats:
        up = worldwide_feats[upstream_key].copy()
        up["_merge_date"] = pd.to_datetime(up["time"]).dt.normalize()
        up = up.drop(columns=["time"])
        df = df.merge(up, on="_merge_date", how="left")

    df = df.drop(columns=["_merge_date"])

    n_new = len([c for c in df.columns if c.startswith(("nao_", "siberian_", "icelandic_",
                 "ns_pressure_", "ecs_pressure_", "natl_pc", "wpac_pc", "up_"))])
    print(f"  Merged {n_new} worldwide features for {region}")
    return df




def build_inference_features(window_dir, region, train_dir):
    """Load and feature-engineer context data for one inference window.

    Same pipeline as training: compute wind speed/direction, pivot to daily (00Z),
    add sub-daily features, lags, rolling stats, elevation, HRES.

    Returns only the **last day** (00Z) of context with all features populated.
    """
    reanalysis_path = window_dir / f"context_reanalysis_{region}.parquet"
    if not reanalysis_path.exists():
        return None

    df = pd.read_parquet(reanalysis_path)
    df["time"] = pd.to_datetime(df["time"])

    # Same pipeline as training
    df = compute_wind_speed_direction(df)
    df = add_temporal_features(df)
    df = pivot_subdaily_features(df, subdaily_vars=["ws10", "wd10", "msl", "t2m"])
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_elevation(df, region, train_dir)

    # Merge HRES surface from inference window directory
    hres_path = window_dir / f"context_hres_{region}.parquet"
    if hres_path.exists():
        hres = pd.read_parquet(hres_path)
        hres["time"] = pd.to_datetime(hres["time"])
        for c in ["latitude", "longitude"]:
            hres[c] = hres[c].round(2)
            df[c] = df[c].round(2)
        fcst_cols = [c for c in hres.columns if c.startswith("fcst_")]
        hres_subset = hres[["time", "latitude", "longitude"] + fcst_cols].copy()
        hres_subset["has_hres"] = 1
        df = df.merge(hres_subset,
                      on=["time", "latitude", "longitude"], how="left")
        df["has_hres"] = df["has_hres"].fillna(0).astype(int)
    else:
        df["has_hres"] = 0

    # Merge HRES pressure-level from inference window directory
    hres_p_path = window_dir / f"context_hres_pressure_{region}.parquet"
    if hres_p_path.exists():
        hres_p = pd.read_parquet(hres_p_path)
        hres_p["time"] = pd.to_datetime(hres_p["time"])
        for c in ["latitude", "longitude"]:
            hres_p[c] = hres_p[c].round(2)
            df[c] = df[c].round(2)
        fcst_p_cols = [c for c in hres_p.columns if c.startswith(("fcst_u_", "fcst_v_"))]
        hres_p_subset = hres_p[["time", "latitude", "longitude"] + fcst_p_cols].copy()
        hres_p_subset["has_hres_pressure"] = 1
        df = df.merge(hres_p_subset,
                      on=["time", "latitude", "longitude"], how="left")
        df["has_hres_pressure"] = df["has_hres_pressure"].fillna(0).astype(int)
    else:
        df["has_hres_pressure"] = 0

    # Mirror training: drop redundant features so inference schema matches
    df = drop_redundant_features(df)

    # Keep only the last day (00Z, all lags populated)
    last_time = df["time"].max()
    df = df[df["time"] == last_time].copy()
    return df



def compute_vertical_ratios(train_dir, region):
    """Compute monthly ws(level)/ws(10m) ratios per grid point.

    Returns a DataFrame: latitude, longitude, month, level, speed_ratio, dir_clim.
    Includes 100m level (from u100/v100) and 5 pressure levels.
    """
    pressure_path = train_dir / f"reanalysis_pressure_{region}.parquet"
    surface_path = train_dir / f"reanalysis_{region}_6h.parquet"

    if not pressure_path.exists():
        print(f"  No pressure data for {region}, will use power-law fallback")
        return None

    print(f"  Loading pressure data: {pressure_path.name}")
    plev = pd.read_parquet(pressure_path)
    plev["time"] = pd.to_datetime(plev["time"])

    surf = pd.read_parquet(surface_path,
                           columns=["time", "latitude", "longitude", "u10", "v10", "u100", "v100"])
    surf["time"] = pd.to_datetime(surf["time"])
    surf["ws10"] = np.sqrt(surf["u10"]**2 + surf["v10"]**2)

    # Aggregate to daily means per grid point
    surf_daily = (surf.groupby([surf["time"].dt.date, "latitude", "longitude"])
                  .agg(ws10=("ws10", "mean")).reset_index())
    surf_daily.rename(columns={"time": "date"}, inplace=True)
    surf_daily["date"] = pd.to_datetime(surf_daily["date"])

    plev["date"] = plev["time"].dt.normalize()

    # Compute wind speed at each pressure level
    for lev in PRESSURE_LEVELS:
        u_col, v_col = f"u_{lev}", f"v_{lev}"
        if u_col in plev.columns and v_col in plev.columns:
            plev[f"ws_{lev}"] = np.sqrt(plev[u_col]**2 + plev[v_col]**2)
            plev[f"wd_{lev}"] = (270 - np.degrees(np.arctan2(plev[v_col], plev[u_col]))) % 360
            plev[f"wd_{lev}_sin"] = np.sin(np.radians(plev[f"wd_{lev}"]))
            plev[f"wd_{lev}_cos"] = np.cos(np.radians(plev[f"wd_{lev}"]))

    ws_cols = [f"ws_{l}" for l in PRESSURE_LEVELS if f"ws_{l}" in plev.columns]
    agg_dict = {c: "mean" for c in ws_cols}
    for l in PRESSURE_LEVELS:
        for suffix in ["_sin", "_cos"]:
            col = f"wd_{l}{suffix}"
            if col in plev.columns:
                agg_dict[col] = "mean"

    plev_daily = (plev.groupby(["date", "latitude", "longitude"])
                  .agg(agg_dict).reset_index())

    for df in [surf_daily, plev_daily]:
        df["latitude"] = df["latitude"].round(2)
        df["longitude"] = df["longitude"].round(2)

    merged = plev_daily.merge(surf_daily, on=["date", "latitude", "longitude"], how="inner")
    merged["month"] = pd.to_datetime(merged["date"]).dt.month

    # Compute ratios per (lat, lon, month, level) — pressure levels
    ratio_rows = []
    grouped = merged.groupby(["latitude", "longitude", "month"])

    for (lat, lon, month), grp in grouped:
        ws10_mean = grp["ws10"].mean()
        if ws10_mean < 0.5:
            continue
        for lev in PRESSURE_LEVELS:
            ws_col = f"ws_{lev}"
            if ws_col not in grp.columns:
                continue
            ratio = grp[ws_col].mean() / ws10_mean
            sin_col, cos_col = f"wd_{lev}_sin", f"wd_{lev}_cos"
            if sin_col in grp.columns:
                dir_clim = np.degrees(np.arctan2(grp[sin_col].mean(), grp[cos_col].mean())) % 360
            else:
                dir_clim = np.nan
            ratio_rows.append({
                "latitude": lat, "longitude": lon, "month": month,
                "level": str(lev), "speed_ratio": ratio, "dir_clim": dir_clim,
            })

    ratios_df = pd.DataFrame(ratio_rows)

    # Add 100m ratios from u100/v100
    if "u100" in surf.columns and "v100" in surf.columns:
        surf["ws100"] = np.sqrt(surf["u100"]**2 + surf["v100"]**2)
        surf["wd100"] = (270 - np.degrees(np.arctan2(surf["v100"], surf["u100"]))) % 360
        surf["wd100_sin"] = np.sin(np.radians(surf["wd100"]))
        surf["wd100_cos"] = np.cos(np.radians(surf["wd100"]))
        surf_100m = (surf.groupby([surf["time"].dt.date, "latitude", "longitude"])
                     .agg(ws10=("ws10", "mean"), ws100=("ws100", "mean"),
                          wd100_sin=("wd100_sin", "mean"),
                          wd100_cos=("wd100_cos", "mean")).reset_index())
        surf_100m.rename(columns={"time": "date"}, inplace=True)
        surf_100m["date"] = pd.to_datetime(surf_100m["date"])
        surf_100m["month"] = surf_100m["date"].dt.month
        for c in ["latitude", "longitude"]:
            surf_100m[c] = surf_100m[c].round(2)
        grp_100m = surf_100m.groupby(["latitude", "longitude", "month"])
        ratio_100m_rows = []
        for (lat, lon, month), grp in grp_100m:
            ws10_mean = grp["ws10"].mean()
            if ws10_mean < 0.5:
                continue
            ratio = grp["ws100"].mean() / ws10_mean
            dir_clim = np.degrees(np.arctan2(
                grp["wd100_sin"].mean(), grp["wd100_cos"].mean())) % 360
            ratio_100m_rows.append({
                "latitude": lat, "longitude": lon, "month": month,
                "level": "100m", "speed_ratio": ratio, "dir_clim": dir_clim,
            })
        if ratio_100m_rows:
            ratios_df = pd.concat([ratios_df, pd.DataFrame(ratio_100m_rows)], ignore_index=True)
            print(f"  Added {len(ratio_100m_rows):,} 100m ratio records")

    print(f"  Computed {len(ratios_df):,} vertical ratio records "
          f"({ratios_df['level'].nunique()} levels, "
          f"{ratios_df.groupby(['latitude','longitude']).ngroups} grid points)")
    return ratios_df


# Compute and save vertical ratios


