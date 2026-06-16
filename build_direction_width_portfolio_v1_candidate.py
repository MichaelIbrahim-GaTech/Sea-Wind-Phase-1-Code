#!/usr/bin/env python3
"""
Rank-aware direction-width portfolio candidate.

Compliance:
- Uses only official files under runs/v6_pressure_speed/phase1_dataset.
- Uses historical train reanalysis/features only for width calibration.
- Uses inference HRES features only to check that a historical HRES coverage
  proxy is close enough to the generated submission centers.
- Does not use external datasets or evaluation target labels.

This is deliberately a width-only branch: it starts from the generated
end-to-end base CSV and changes dir_05/dir_95 around the existing dir_50
centers only for blocks that pass an official historical CV gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sea_winds_end_to_end_final as E2E


ROOT = Path("runs/v6_pressure_speed")
DATA = ROOT / "phase1_dataset"
FEATURES = DATA / "features"
TRAIN = DATA / "train"

DEFAULT_BASE_CSV = ROOT / "pred_ns_p14dir_hres.csv"
OUT_CSV = ROOT / "pred_dir_width_portfolio_v1.csv"
OUT_ZIP = ROOT / "sub_dirw_port_v1.zip"
SUMMARY_CSV = ROOT / "cv_dir_width_portfolio_v1.csv"
MANIFEST = ROOT / "manifest_dir_width_portfolio_v1.json"

REGIONS = ("north_sea", "east_china_sea")
GROUP_LEVELS = {
    "surface": ("10m", "100m"),
    "pressure": ("1000", "925", "850", "700", "500"),
}
HOURS = (0, 6, 12, 18)
HORIZONS = (7, 14)
WIDTH_GRID = np.array(list(np.arange(80.0, 170.1, 2.5)) + [179.9], dtype="float64")

# A width change must clear these official-data gates before it can touch the
# submission. The public leaderboard is not used here; every gate is based on
# historical train/features/reanalysis plus inference center-proxy alignment.
MIN_MEAN_GAIN = 5.0
MAX_ALLOWED_WORSE_FOLD = 1.0
MIN_WIDTH_CHANGE = 5.0
MAX_CENTER_PROXY_MEDIAN_DIFF = 25.0
VAL_YEARS = (2020, 2021)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_names(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def hres_lead(horizon: int) -> int:
    return horizon if horizon in (1, 7) else 10


def speed_dir_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype="float64")
    v = np.asarray(v, dtype="float64")
    speed = np.sqrt(u * u + v * v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed.astype("float32"), direction.astype("float32")


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def circular_winkler_score(y: np.ndarray, center: np.ndarray, half_width: float) -> float:
    y = np.asarray(y, dtype="float64") % 360.0
    center = np.asarray(center, dtype="float64") % 360.0
    ok = np.isfinite(y) & np.isfinite(center)
    if not bool(ok.any()):
        return float("nan")
    y = y[ok]
    center = center[ok]
    lo = (center - half_width) % 360.0
    hi = (center + half_width) % 360.0
    width = (hi - lo) % 360.0
    inside = ((y - lo) % 360.0) <= width
    miss = np.minimum(circ_abs_diff(y, lo), circ_abs_diff(y, hi))
    return float(np.mean(width + 20.0 * miss * (~inside)))


def best_width(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    best_score = float("inf")
    best_hw = float("nan")
    for hw in WIDTH_GRID:
        score = circular_winkler_score(y, center, float(hw))
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    return best_score, best_hw


def inference_origins() -> list[pd.Timestamp]:
    origins = []
    for window in range(1, 9):
        meta = json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text())
        origins.append(pd.Timestamp(meta["context_end"]))
    return origins


@dataclass
class CubeStore:
    n_grid: int
    latlon: pd.DataFrame
    time_to_idx: dict[pd.Timestamp, int]
    u: dict[str, np.ndarray]
    v: dict[str, np.ndarray]


def load_cube(region: str, group: str) -> CubeStore:
    if group == "surface":
        path = TRAIN / f"reanalysis_{region}_6h.parquet"
        cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
        level_cols = {"10m": ("u10", "v10"), "100m": ("u100", "v100")}
    else:
        path = TRAIN / f"reanalysis_pressure_{region}.parquet"
        cols = ["time", "latitude", "longitude"]
        for level in GROUP_LEVELS["pressure"]:
            cols += [f"u_{level}", f"v_{level}"]
        level_cols = {level: (f"u_{level}", f"v_{level}") for level in GROUP_LEVELS["pressure"]}

    df = pd.read_parquet(path, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    df = df.sort_values(["time", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
    times = pd.Series(df["time"].unique()).sort_values().to_numpy()
    n_times = len(times)
    n_grid = int(len(df) // n_times)
    latlon = df.loc[: n_grid - 1, ["latitude", "longitude"]].reset_index(drop=True)

    u_arrays: dict[str, np.ndarray] = {}
    v_arrays: dict[str, np.ndarray] = {}
    for level, (u_col, v_col) in level_cols.items():
        u_arrays[level] = df[u_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
        v_arrays[level] = df[v_col].to_numpy(dtype="float32").reshape(n_times, n_grid)
    return CubeStore(
        n_grid=n_grid,
        latlon=latlon,
        time_to_idx={pd.Timestamp(t): i for i, t in enumerate(times)},
        u=u_arrays,
        v=v_arrays,
    )


def hres_columns(group: str, horizon: int) -> list[str]:
    lead = hres_lead(horizon)
    cols = ["time", "latitude", "longitude"]
    for hour in HOURS:
        if group == "surface":
            cols += [f"fcst_speed_d{lead}_h{hour}", f"fcst_dir_d{lead}_h{hour}"]
        else:
            for level in GROUP_LEVELS[group]:
                cols += [f"fcst_u_{level}_d{lead}_h{hour}", f"fcst_v_{level}_d{lead}_h{hour}"]
    return list(dict.fromkeys(cols))


def load_feature_df(region: str, group: str, horizon: int) -> pd.DataFrame:
    path = FEATURES / f"train_{region}.parquet"
    available = set(schema_names(path))
    cols = [c for c in hres_columns(group, horizon) if c in available]
    df = pd.read_parquet(path, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype("float32").round(2)
    df["longitude"] = df["longitude"].astype("float32").round(2)
    return df.reset_index(drop=True)


def attach_grid_index(df: pd.DataFrame, cube: CubeStore, region: str, group: str) -> pd.DataFrame:
    grid = cube.latlon.reset_index().rename(columns={"index": "grid_idx"})
    out = df.merge(grid, on=["latitude", "longitude"], how="left", sort=False)
    if out["grid_idx"].isna().any():
        raise RuntimeError(f"{region}/{group} feature rows missing target grid_idx")
    out["grid_idx"] = out["grid_idx"].astype("int32")
    return out


def hres_dir_from_rows(df: pd.DataFrame, row_idx: np.ndarray, group: str, level: str, horizon: int, hour: int) -> np.ndarray:
    lead = hres_lead(horizon)
    if group == "surface":
        return pd.to_numeric(df.iloc[row_idx][f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0
    u = pd.to_numeric(df.iloc[row_idx][f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df.iloc[row_idx][f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
    return speed_dir_from_uv(u, v)[1].astype("float64")


def target_dir(cube: CubeStore, origin_times: np.ndarray, grid_idx: np.ndarray, level: str, horizon: int, hour: int) -> np.ndarray:
    future = pd.to_datetime(origin_times) + pd.to_timedelta(horizon, unit="D") + pd.to_timedelta(hour, unit="h")
    t_idx = np.array([cube.time_to_idx.get(pd.Timestamp(t), -1) for t in future], dtype="int32")
    ok = t_idx >= 0
    out = np.full(len(grid_idx), np.nan, dtype="float64")
    if bool(ok.any()):
        u = cube.u[level][t_idx[ok], grid_idx[ok]].astype("float64")
        v = cube.v[level][t_idx[ok], grid_idx[ok]].astype("float64")
        out[ok] = speed_dir_from_uv(u, v)[1]
    return out


def collect_validation(
    region: str,
    group: str,
    horizon: int,
    grid_per_anchor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = load_cube(region, group)
    df = attach_grid_index(load_feature_df(region, group, horizon), cube, region, group)
    rng = np.random.default_rng(seed + horizon * 97 + (0 if group == "surface" else 1000) + (0 if region == "north_sea" else 2000))
    y_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    anchors = inference_origins()

    for val_year in VAL_YEARS:
        for anchor in anchors:
            origin = pd.Timestamp(year=val_year, month=anchor.month, day=anchor.day)
            rows = np.flatnonzero(df["time"].eq(origin).to_numpy())
            if len(rows) == 0:
                continue
            if len(rows) > grid_per_anchor:
                rows = np.sort(rng.choice(rows, size=grid_per_anchor, replace=False))
            grid_idx = df.iloc[rows]["grid_idx"].to_numpy(dtype="int32")
            origin_times = df.iloc[rows]["time"].to_numpy()
            for level in GROUP_LEVELS[group]:
                for hour in HOURS:
                    center = hres_dir_from_rows(df, rows, group, level, horizon, hour)
                    actual = target_dir(cube, origin_times, grid_idx, level, horizon, hour)
                    ok = np.isfinite(center) & np.isfinite(actual)
                    if not bool(ok.any()):
                        continue
                    y_parts.append(actual[ok])
                    c_parts.append(center[ok])
                    fold_parts.append(np.full(int(ok.sum()), val_year, dtype="int16"))

    if not y_parts:
        raise RuntimeError(f"no validation rows for {region}/{group}/d{horizon}")
    return np.concatenate(y_parts), np.concatenate(c_parts), np.concatenate(fold_parts)


def current_half_widths(base_csv: Path) -> dict[tuple[str, str, int], float]:
    usecols = ["type", "region", "horizon", "level", "dir_05", "dir_95"]
    parts = []
    for chunk in pd.read_csv(base_csv, usecols=usecols, chunksize=600_000):
        grid = chunk[chunk["type"].eq("grid")].copy()
        grid["group"] = np.where(grid["level"].isin(GROUP_LEVELS["surface"]), "surface", "pressure")
        grid["half_width"] = ((grid["dir_95"] - grid["dir_05"]) % 360.0) / 2.0
        parts.append(grid.groupby(["region", "group", "horizon"])["half_width"].median())
    s = pd.concat(parts).groupby(level=[0, 1, 2]).median()
    return {(str(r), str(g), int(h)): float(v) for (r, g, h), v in s.items()}


def inference_hres_centers(region: str, group: str, horizon: int) -> pd.DataFrame:
    lead = hres_lead(horizon)
    rows = []
    cols = ["latitude", "longitude"]
    for hour in HOURS:
        if group == "surface":
            cols += [f"fcst_dir_d{lead}_h{hour}"]
        else:
            for level in GROUP_LEVELS[group]:
                cols += [f"fcst_u_{level}_d{lead}_h{hour}", f"fcst_v_{level}_d{lead}_h{hour}"]
    cols = list(dict.fromkeys(cols))

    for window in range(1, 9):
        inf = pd.read_parquet(FEATURES / f"inference_window_{window}_{region}.parquet", columns=cols)
        inf["latitude"] = inf["latitude"].astype("float32").round(2)
        inf["longitude"] = inf["longitude"].astype("float32").round(2)
        for hour in HOURS:
            if group == "surface":
                part = inf[["latitude", "longitude"]].copy()
                part["level"] = "10m"
                part["center"] = pd.to_numeric(inf[f"fcst_dir_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64") % 360.0
                for level in GROUP_LEVELS["surface"]:
                    tmp = part.copy()
                    tmp["level"] = level
                    tmp["window"] = window
                    tmp["hour"] = hour
                    rows.append(tmp)
            else:
                for level in GROUP_LEVELS["pressure"]:
                    u = pd.to_numeric(inf[f"fcst_u_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                    v = pd.to_numeric(inf[f"fcst_v_{level}_d{lead}_h{hour}"], errors="coerce").to_numpy(dtype="float64")
                    part = inf[["latitude", "longitude"]].copy()
                    part["level"] = level
                    part["center"] = speed_dir_from_uv(u, v)[1].astype("float64")
                    part["window"] = window
                    part["hour"] = hour
                    rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["region"] = region
    out["horizon"] = horizon
    return out[["window", "region", "latitude", "longitude", "horizon", "hour", "level", "center"]]


def center_proxy_alignment(base_csv: Path, region: str, group: str, horizon: int) -> tuple[float, float]:
    levels = set(GROUP_LEVELS[group])
    hres = inference_hres_centers(region, group, horizon)
    diffs = []
    usecols = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level", "dir_50"]
    for chunk in pd.read_csv(base_csv, usecols=usecols, chunksize=600_000):
        m = (
            chunk["type"].eq("grid")
            & chunk["region"].eq(region)
            & chunk["horizon"].eq(horizon)
            & chunk["level"].isin(levels)
        )
        if not bool(m.any()):
            continue
        cur = chunk.loc[m].copy()
        cur["latitude"] = pd.to_numeric(cur["latitude"], errors="coerce").round(2)
        cur["longitude"] = pd.to_numeric(cur["longitude"], errors="coerce").round(2)
        cur["dir_50"] = pd.to_numeric(cur["dir_50"], errors="coerce") % 360.0
        merged = cur.merge(
            hres,
            on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
            how="left",
            validate="many_to_one",
        )
        if merged["center"].isna().any():
            raise RuntimeError(f"missing inference HRES centers for {region}/{group}/d{horizon}")
        diffs.append(circ_abs_diff(merged["dir_50"].to_numpy(dtype="float64"), merged["center"].to_numpy(dtype="float64")))
    if not diffs:
        raise RuntimeError(f"no base rows for {region}/{group}/d{horizon}")
    arr = np.concatenate(diffs)
    return float(np.nanmedian(arr)), float(np.nanmean(arr))


def evaluate_blocks(base_csv: Path, grid_per_anchor: int, seed: int) -> pd.DataFrame:
    cur_hw = current_half_widths(base_csv)
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        for group in GROUP_LEVELS:
            for horizon in HORIZONS:
                key = (region, group, horizon)
                log(f"[cv] {region}/{group}/d{horizon}")
                y, center, fold = collect_validation(region, group, horizon, grid_per_anchor, seed)
                current_hw = cur_hw[key]
                current_scores = []
                best_scores = []
                best_all_score, selected_hw = best_width(y, center)
                for val_year in VAL_YEARS:
                    m = fold == val_year
                    current_scores.append(circular_winkler_score(y[m], center[m], current_hw))
                    best_scores.append(circular_winkler_score(y[m], center[m], selected_hw))
                current_mean = float(np.mean(current_scores))
                selected_mean = float(np.mean(best_scores))
                fold_gains = np.asarray(current_scores) - np.asarray(best_scores)
                proxy_median, proxy_mean = center_proxy_alignment(base_csv, region, group, horizon)
                gate_passed = (
                    selected_mean <= current_mean - MIN_MEAN_GAIN
                    and float(np.min(fold_gains)) >= -MAX_ALLOWED_WORSE_FOLD
                    and abs(selected_hw - current_hw) >= MIN_WIDTH_CHANGE
                    and proxy_median <= MAX_CENTER_PROXY_MEDIAN_DIFF
                )
                rows.append(
                    {
                        "region": region,
                        "group": group,
                        "horizon": horizon,
                        "n": int(len(y)),
                        "current_hw": current_hw,
                        "selected_hw": selected_hw,
                        "cv_current_mean": current_mean,
                        "cv_selected_mean": selected_mean,
                        "cv_mean_gain": current_mean - selected_mean,
                        "cv_min_fold_gain": float(np.min(fold_gains)),
                        "cv_max_fold_gain": float(np.max(fold_gains)),
                        "center_proxy_median_abs_diff": proxy_median,
                        "center_proxy_mean_abs_diff": proxy_mean,
                        "gate_passed": bool(gate_passed),
                    }
                )
    return pd.DataFrame(rows)


def apply_widths(base_csv: Path, summary: pd.DataFrame, output_csv: Path, output_zip: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    selected = summary[summary["gate_passed"].astype(bool)].copy()
    df = pd.read_csv(base_csv, low_memory=False)
    counts: dict[str, int] = {}
    for _, row in selected.iterrows():
        region = str(row["region"])
        group = str(row["group"])
        horizon = int(row["horizon"])
        half_width = float(row["selected_hw"])
        levels = set(GROUP_LEVELS[group])
        m = (
            df["type"].eq("grid")
            & df["region"].eq(region)
            & df["horizon"].eq(horizon)
            & df["level"].isin(levels)
        )
        idx = df.index[m]
        center = pd.to_numeric(df.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        df.loc[idx, "dir_05"] = (center - half_width) % 360.0
        df.loc[idx, "dir_95"] = (center + half_width) % 360.0
        counts[f"{region}_{group}_d{horizon}_hw{half_width:g}"] = int(len(idx))

    final = E2E.validate_final(df)
    E2E.write_zip(final, output_csv, output_zip)
    return final, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--output-zip", type=Path, default=OUT_ZIP)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--grid-per-anchor", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    if not args.base_csv.exists():
        raise SystemExit(f"missing base CSV: {args.base_csv}")
    if args.output_zip.name.__len__() >= 64:
        raise SystemExit(f"zip filename is too long for Codabench: {args.output_zip.name}")

    log("Direction-width portfolio v1")
    log(f"Base CSV: {args.base_csv} ({args.base_csv.stat().st_size:,} bytes)")
    summary = evaluate_blocks(args.base_csv, args.grid_per_anchor, args.seed)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    log(f"Wrote CV summary: {args.summary_csv}")
    log(summary.sort_values("cv_mean_gain", ascending=False).to_string(index=False))

    selected = summary[summary["gate_passed"].astype(bool)]
    if selected.empty:
        payload = {
            "status": "gate_failed_no_submission_written",
            "reason": "No direction-width block passed the official historical CV and center-proxy gates.",
            "base_csv": str(args.base_csv.resolve()),
            "cv_summary_csv": str(args.summary_csv.resolve()),
            "compliance": {
                "external_training_data_used": False,
                "web_data_used": False,
                "evaluation_target_labels_used_for_training": False,
                "official_dataset_root": str(DATA.resolve()),
            },
        }
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("No gated width portfolio candidate was emitted.")

    final, counts = apply_widths(args.base_csv, summary, args.output_csv, args.output_zip)
    payload = {
        "status": "completed",
        "base_csv": {
            "path": str(args.base_csv.resolve()),
            "size": args.base_csv.stat().st_size,
            "sha256": sha256(args.base_csv),
        },
        "cv_summary_csv": str(args.summary_csv.resolve()),
        "selected_blocks": selected.to_dict(orient="records"),
        "patch_counts": counts,
        "submission": {
            "csv": str(args.output_csv.resolve()),
            "csv_size": args.output_csv.stat().st_size,
            "csv_sha256": sha256(args.output_csv),
            "zip": str(args.output_zip.resolve()),
            "zip_size": args.output_zip.stat().st_size,
            "zip_name_length": len(args.output_zip.name),
            "zip_sha256": sha256(args.output_zip),
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts().to_dict().items()},
        },
        "compliance": {
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "official_dataset_root": str(DATA.resolve()),
            "notes": [
                "Width gates use only official historical HRES features and reanalysis targets.",
                "Inference center-proxy alignment uses official inference HRES features but no evaluation targets.",
                "Final patch changes only dir_05/dir_95 around the generated end-to-end dir_50 centers.",
            ],
        },
        "code_hashes": {
            "build_direction_width_portfolio_v1_candidate.py": sha256(Path(__file__).resolve()),
            "sea_winds_end_to_end_final.py": sha256(Path("sea_winds_end_to_end_final.py")),
        },
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {args.manifest}")
    log(f"OK: {args.output_zip}")


if __name__ == "__main__":
    main()
