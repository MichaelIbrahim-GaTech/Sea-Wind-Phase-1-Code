from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_sfc14_selw.csv"
BACKTEST_CSV = WORK / "seasonal_direction_backtest_on_sfc14_selw.csv"
OUT_CSV = WORK / "pred_seasbt_gate.csv"
OUT_ZIP = WORK / "sub_seasbt_gate.zip"
MANIFEST = WORK / "manifest_seasbt_gate.json"

HOURS = (0, 6, 12, 18)
GROUP_LEVELS = {
    "surface": ("10m", "100m"),
    "pressure": ("1000", "925", "850", "700", "500"),
}
WINDOWS = (7, 14, 21, 30, 45, 60)

PUBLIC_CURRENT = {
    ("north_sea", "surface", 7): 298.5943,
    ("north_sea", "surface", 14): 338.5514,
    ("north_sea", "pressure", 7): 285.4723,
    ("north_sea", "pressure", 14): 330.5092,
    ("east_china_sea", "surface", 7): 278.4071,
    ("east_china_sea", "surface", 14): 327.5396,
    ("east_china_sea", "pressure", 7): 252.7414,
    ("east_china_sea", "pressure", 14): 315.4798,
}
PUBLIC_TOP5_BEST = {
    ("north_sea", "surface", 7): 256.44,
    ("north_sea", "surface", 14): 307.42,
    ("north_sea", "pressure", 7): 236.54,
    ("north_sea", "pressure", 14): 300.28,
    ("east_china_sea", "surface", 7): 265.19,
    ("east_china_sea", "surface", 14): 305.01,
    ("east_china_sea", "pressure", 7): 216.52,
    ("east_china_sea", "pressure", 14): 288.19,
}

MIN_BACKTEST_GAIN = 7.5
MIN_PUBLIC_GAP = 8.0

COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS


def log(msg: str) -> None:
    print(msg, flush=True)


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


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP, MANIFEST):
        if path.exists():
            path.unlink()


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    for c in SPEED_COLS + DIR_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def circ_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0) - 180.0)


def circular_mean(parts: list[np.ndarray]) -> np.ndarray:
    arr = np.vstack([np.asarray(p, dtype="float64") % 360.0 for p in parts])
    valid = np.isfinite(arr)
    rad = np.deg2rad(np.where(valid, arr, 0.0))
    sin = np.sum(np.sin(rad) * valid, axis=0)
    cos = np.sum(np.cos(rad) * valid, axis=0)
    out = np.degrees(np.arctan2(sin, cos)) % 360.0
    out[(np.abs(sin) + np.abs(cos)) <= 1e-12] = np.nan
    return out


def blend_direction(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    ar = np.deg2rad(np.asarray(a, dtype="float64") % 360.0)
    br = np.deg2rad(np.asarray(b, dtype="float64") % 360.0)
    x = (1.0 - weight_b) * np.cos(ar) + weight_b * np.cos(br)
    y = (1.0 - weight_b) * np.sin(ar) + weight_b * np.sin(br)
    out = np.degrees(np.arctan2(y, x)) % 360.0
    out[~np.isfinite(a) | ~np.isfinite(b)] = np.nan
    return out


def parse_candidate(candidate: str) -> tuple[int, float | None]:
    if candidate.startswith("seasonal_w"):
        return int(candidate.split("w", 1)[1]), None
    parts = candidate.split("_")
    return int(parts[2][1:]), float(parts[3])


def select_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(BACKTEST_CSV, "Run seasonal_direction_backtest.py against pred_sfc14_selw.csv first.")
    df = pd.read_csv(BACKTEST_CSV)
    current = df[df["candidate"].eq("current_model")][
        ["region", "group", "horizon", "score", "half_width"]
    ].rename(columns={"score": "current_score", "half_width": "current_half_width"})
    cand = df[~df["candidate"].eq("current_model")].merge(
        current, on=["region", "group", "horizon"], how="left", validate="many_to_one"
    )
    cand["gain_vs_current"] = cand["current_score"] - cand["score"]
    public_gaps = []
    gate = []
    for row in cand.itertuples(index=False):
        key = (str(row.region), str(row.group), int(row.horizon))
        gap = float(PUBLIC_CURRENT[key] - PUBLIC_TOP5_BEST[key])
        public_gaps.append(gap)
        gate.append(bool(gap >= MIN_PUBLIC_GAP and float(row.gain_vs_current) >= MIN_BACKTEST_GAIN))
    cand["public_gap"] = public_gaps
    cand["gate_passed"] = gate
    cand = cand.sort_values(
        ["gate_passed", "region", "group", "horizon", "gain_vs_current", "score"],
        ascending=[False, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = cand[cand["gate_passed"].astype(bool)].groupby(["region", "group", "horizon"], as_index=False).head(1)
    selected = selected.reset_index(drop=True)
    return cand, selected


def load_actual(region: str, group: str) -> pd.DataFrame:
    if group == "surface":
        cols = ["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]
        df = pd.read_parquet(DATA / "train" / f"reanalysis_{region}_6h.parquet", columns=cols)
        df["time"] = pd.to_datetime(df["time"])
        df["latitude"] = df["latitude"].astype(float).round(2)
        df["longitude"] = df["longitude"].astype(float).round(2)
        df["year"] = df["time"].dt.year.astype("int16")
        df["doy"] = df["time"].dt.dayofyear.astype("int16")
        df["hour"] = df["time"].dt.hour.astype("int8")
        df["dir_10m"] = (270.0 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360.0
        df["dir_100m"] = (270.0 - np.degrees(np.arctan2(df["v100"], df["u100"]))) % 360.0
        return df[["time", "latitude", "longitude", "year", "doy", "hour", "dir_10m", "dir_100m"]]

    cols = ["time", "latitude", "longitude"]
    for level in GROUP_LEVELS["pressure"]:
        cols.extend([f"u_{level}", f"v_{level}"])
    df = pd.read_parquet(DATA / "train" / f"reanalysis_pressure_{region}.parquet", columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = df["latitude"].astype(float).round(2)
    df["longitude"] = df["longitude"].astype(float).round(2)
    df["year"] = df["time"].dt.year.astype("int16")
    df["doy"] = df["time"].dt.dayofyear.astype("int16")
    df["hour"] = df["time"].dt.hour.astype("int8")
    keep = ["time", "latitude", "longitude", "year", "doy", "hour"]
    for level in GROUP_LEVELS["pressure"]:
        df[f"dir_{level}"] = (270.0 - np.degrees(np.arctan2(df[f"v_{level}"], df[f"u_{level}"]))) % 360.0
        keep.append(f"dir_{level}")
    return df[keep]


def doy_distance(series: pd.Series, doy: int) -> pd.Series:
    d = (series.astype(int) - int(doy)).abs()
    return np.minimum(d, 366 - d)


def seasonal_centers(
    hist_by_hour: dict[int, pd.DataFrame],
    level: str,
    target_time: pd.Timestamp,
    coords: pd.DataFrame,
    window_days: int,
) -> np.ndarray:
    subset = hist_by_hour.get(int(target_time.hour))
    if subset is None or subset.empty:
        return np.full(len(coords), np.nan)
    keep = doy_distance(subset["doy"], int(target_time.dayofyear)) <= int(window_days)
    tmp = subset.loc[keep, ["latitude", "longitude", f"dir_{level}"]].copy()
    if tmp.empty:
        return np.full(len(coords), np.nan)
    ang = np.deg2rad(pd.to_numeric(tmp[f"dir_{level}"], errors="coerce").to_numpy(dtype="float64") % 360.0)
    tmp["sin"] = np.sin(ang)
    tmp["cos"] = np.cos(ang)
    grp = tmp.groupby(["latitude", "longitude"], sort=False)[["sin", "cos"]].mean().reset_index()
    grp["seasonal_dir"] = np.degrees(np.arctan2(grp["sin"], grp["cos"])) % 360.0
    merged = coords.merge(grp[["latitude", "longitude", "seasonal_dir"]], on=["latitude", "longitude"], how="left")
    return merged["seasonal_dir"].to_numpy(dtype="float64")


def window_metadata(window: int) -> dict[str, object]:
    return json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text(encoding="utf-8"))


def apply_selected(base: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, object]]]:
    patched = base.copy()
    actual_cache: dict[tuple[str, str], pd.DataFrame] = {}
    hist_cache: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
    patch_counts: dict[str, int] = {}
    stability: list[dict[str, object]] = []

    for row in selected.itertuples(index=False):
        region = str(row.region)
        group = str(row.group)
        horizon = int(row.horizon)
        candidate = str(row.candidate)
        window_days, weight_current = parse_candidate(candidate)
        half_width = float(np.clip(round(float(row.half_width) / 5.0) * 5.0, 15.0, 179.9))
        actual = actual_cache.setdefault((region, group), load_actual(region, group))
        hist_by_hour = hist_cache.setdefault(
            (region, group),
            {int(hour): part.reset_index(drop=True) for hour, part in actual[actual["year"].le(2021)].groupby("hour", sort=False)},
        )
        deltas = []
        changed = 0
        log(f"Applying {region}/{group}/d{horizon}: {candidate}, half_width={half_width:g}")
        for window in range(1, 9):
            meta = window_metadata(window)
            score_day = pd.Timestamp(meta["score_days"][f"d{horizon}"])
            for hour in HOURS:
                target_time = score_day + pd.Timedelta(hours=int(hour))
                for level in GROUP_LEVELS[group]:
                    idx = patched.index[
                        patched["type"].eq("grid")
                        & patched["region"].eq(region)
                        & patched["window"].eq(window)
                        & patched["horizon"].eq(horizon)
                        & patched["hour"].eq(hour)
                        & patched["level"].eq(level)
                    ]
                    if len(idx) == 0:
                        raise SystemExit(f"missing rows for {region}/{group}/d{horizon}/w{window}/h{hour}/{level}")
                    coords = patched.loc[idx, ["latitude", "longitude"]].copy()
                    seasonal = seasonal_centers(hist_by_hour, level, target_time, coords, window_days)
                    if np.isnan(seasonal).any():
                        raise SystemExit(f"seasonal center has missing values for {region}/{group}/d{horizon}/w{window}/h{hour}/{level}")
                    current = pd.to_numeric(patched.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
                    center = seasonal if weight_current is None else blend_direction(seasonal, current, float(weight_current))
                    deltas.append(circ_abs_diff(center, current))
                    patched.loc[idx, "dir_50"] = center % 360.0
                    patched.loc[idx, "dir_05"] = (center - half_width) % 360.0
                    patched.loc[idx, "dir_95"] = (center + half_width) % 360.0
                    changed += int(len(idx))
        d = np.concatenate(deltas)
        block_name = f"{region}_{group}_d{horizon}_{candidate}"
        patch_counts[block_name] = int(changed)
        stability.append(
            {
                "block": block_name,
                "rows": int(changed),
                "half_width": half_width,
                "center_delta_mean": float(np.nanmean(d)),
                "center_delta_p50": float(np.nanquantile(d, 0.50)),
                "center_delta_p90": float(np.nanquantile(d, 0.90)),
                "center_delta_p99": float(np.nanquantile(d, 0.99)),
            }
        )
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


def validate_delta(before: pd.DataFrame, after: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    speed_changed = rows_changed(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, after, DIR_COLS, 1, circular=True)
    allowed = np.zeros(len(after), dtype=bool)
    rows_by_block = {}
    for row in selected.itertuples(index=False):
        region = str(row.region)
        group = str(row.group)
        horizon = int(row.horizon)
        mask = (
            after["type"].eq("grid")
            & after["region"].eq(region)
            & after["horizon"].eq(horizon)
            & after["level"].isin(GROUP_LEVELS[group])
        ).to_numpy(dtype=bool)
        allowed |= mask
        rows_by_block[f"{region}_{group}_d{horizon}"] = int(mask.sum())
    outside = dir_changed & ~allowed
    if int(speed_changed.sum()) != 0:
        raise SystemExit(f"unexpected speed rows changed: {int(speed_changed.sum())}")
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected direction rows outside selected blocks: {int(outside.sum())}")
    return {
        "target_rows_by_block": rows_by_block,
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_direction_rows_changed": int(outside.sum()),
    }


def zip_payload() -> dict[str, object]:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")
    return {
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
    }


def write_manifest(
    status: str,
    reason: str,
    decision_table: pd.DataFrame,
    selected: pd.DataFrame,
    patch_counts: dict[str, int] | None = None,
    stability: list[dict[str, object]] | None = None,
    delta: dict[str, object] | None = None,
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "submission": zip_payload() if OUT_ZIP.exists() else None,
        "base_csv": {
            "path": str(BASE_CSV),
            "size": int(BASE_CSV.stat().st_size) if BASE_CSV.exists() else None,
            "sha256": sha256(BASE_CSV) if BASE_CSV.exists() else None,
        },
        "backtest_csv": str(BACKTEST_CSV),
        "decision_table_head": decision_table.head(80).to_dict(orient="records"),
        "selected_blocks": selected.to_dict(orient="records"),
        "patch_counts": patch_counts or {},
        "inference_stability": stability or [],
        "delta": delta or {},
        "gates": {
            "min_backtest_gain": MIN_BACKTEST_GAIN,
            "min_public_gap": MIN_PUBLIC_GAP,
            "public_metrics_used_only_for_rank_risk": True,
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "notes": [
                "Backtest table is generated from official historical train reanalysis and the generated current-best prediction file.",
                "Final inference seasonal centers use official 2019-2021 train reanalysis, all earlier than the 2022 inference windows.",
                "Public leaderboard metrics are used only as fixed rank-risk thresholds, not model features or labels.",
                "No external datasets or evaluation target labels are read.",
            ],
        },
        "code_hashes": {
            "build_seasonal_backtest_gate_candidate.py": sha256(Path(__file__).resolve()),
            "seasonal_direction_backtest.py": sha256(ROOT / "seasonal_direction_backtest.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(payload, indent=2, sort_keys=True))
    log(f"Wrote {MANIFEST}")


def main() -> None:
    require(BASE_CSV, "Run .\\run_surface_d14_width_on_selective_center_e2e.ps1 first.")
    cleanup_outputs()
    decision_table, selected = select_blocks()
    if selected.empty:
        write_manifest(
            "gate_failed_no_submission_written",
            "No seasonal backtest candidate cleared the gain and rank-risk gates.",
            decision_table,
            selected,
        )
        return

    log("Selected seasonal backtest blocks:")
    log(selected[["region", "group", "horizon", "candidate", "gain_vs_current", "score", "half_width"]].to_string(index=False))
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    before = base.copy()
    patched, patch_counts, stability = apply_selected(base, selected)
    final = E2E.validate_final(patched)
    delta = validate_delta(before, final, selected)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(
        "submission_written_after_seasonal_backtest_gate",
        "Seasonal backtest gain, rank-risk, schema, and zip gates passed.",
        decision_table,
        selected,
        patch_counts=patch_counts,
        stability=stability,
        delta=delta,
    )
    log(f"OK: {OUT_ZIP}")


if __name__ == "__main__":
    main()
