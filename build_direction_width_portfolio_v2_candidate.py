#!/usr/bin/env python3
"""
Fine-grained direction-width portfolio candidate.

Compliance:
- Uses only official files under runs/v6_pressure_speed/phase1_dataset.
- Uses historical train reanalysis/features for chronological width CV.
- Uses official inference HRES features only for center-proxy alignment checks.
- Does not use external datasets or evaluation target labels.

This branch is width-only. It starts from the current generated base CSV and
changes dir_05/dir_95 around existing dir_50 centers only for blocks that pass
strict historical CV gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_direction_width_portfolio_v1_candidate as V1
import sea_winds_end_to_end_final as E2E


ROOT = V1.ROOT
DATA = V1.DATA
DEFAULT_BASE_CSV = ROOT / "pred_ns_p7dir_mosres.csv"
OUT_CSV = ROOT / "pred_dir_width_portfolio_v2.csv"
OUT_ZIP = ROOT / "sub_dirw_port_v2.zip"
SUMMARY_CSV = ROOT / "cv_dir_width_portfolio_v2.csv"
MANIFEST = ROOT / "manifest_dir_width_portfolio_v2.json"

REGIONS = V1.REGIONS
GROUP_LEVELS = V1.GROUP_LEVELS
HOURS = V1.HOURS
HORIZONS = V1.HORIZONS
VAL_YEARS = V1.VAL_YEARS
WIDTH_GRID = np.array(list(np.arange(70.0, 170.1, 2.5)) + [172.5, 175.0, 177.5, 179.9], dtype="float64")

MIN_N = 2500
MIN_WIDTH_CHANGE = 5.0
MAX_ALLOWED_WORSE_FOLD = 0.5
MAX_CENTER_PROXY_MEDIAN_DIFF = 60.0
MAX_CENTER_PROXY_MEAN_DIFF = 75.0
MIN_MEAN_GAIN = {
    "group": 7.0,
    "level": 5.0,
    "hour": 5.0,
    "level_hour": 6.0,
}
SPECIFICITY = {"group": 0, "hour": 1, "level": 2, "level_hour": 3}


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_key(mode: str, region: str, group: str, horizon: int, level: str = "*", hour: int | str = "*") -> tuple[str, str, str, int, str, str]:
    return (mode, str(region), str(group), int(horizon), str(level), str(hour))


def key_record(key: tuple[str, str, str, int, str, str]) -> dict[str, object]:
    mode, region, group, horizon, level, hour = key
    return {
        "mode": mode,
        "region": region,
        "group": group,
        "horizon": int(horizon),
        "level": level,
        "hour": hour,
        "specificity": int(SPECIFICITY[mode]),
    }


def best_width(y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    best_score = float("inf")
    best_hw = float("nan")
    for hw in WIDTH_GRID:
        score = V1.circular_winkler_score(y, center, float(hw))
        if score < best_score:
            best_score = score
            best_hw = float(hw)
    return best_score, best_hw


def append_block(
    store: dict[tuple[str, str, str, int, str, str], dict[str, list[np.ndarray]]],
    key: tuple[str, str, str, int, str, str],
    actual: np.ndarray,
    center: np.ndarray,
    fold: np.ndarray,
) -> None:
    if key not in store:
        store[key] = {"y": [], "center": [], "fold": []}
    store[key]["y"].append(actual)
    store[key]["center"].append(center)
    store[key]["fold"].append(fold)


def collect_validation_blocks(region: str, group: str, horizon: int, grid_per_anchor: int, seed: int) -> dict:
    cube = V1.load_cube(region, group)
    df = V1.attach_grid_index(V1.load_feature_df(region, group, horizon), cube, region, group)
    rng = np.random.default_rng(seed + horizon * 97 + (0 if group == "surface" else 1000) + (0 if region == "north_sea" else 2000))
    anchors = V1.inference_origins()
    store: dict[tuple[str, str, str, int, str, str], dict[str, list[np.ndarray]]] = {}

    for val_year in VAL_YEARS:
        for anchor in anchors:
            origin = pd.Timestamp(year=int(val_year), month=anchor.month, day=anchor.day)
            rows = np.flatnonzero(df["time"].eq(origin).to_numpy())
            if len(rows) == 0:
                continue
            if len(rows) > grid_per_anchor:
                rows = np.sort(rng.choice(rows, size=grid_per_anchor, replace=False))
            grid_idx = df.iloc[rows]["grid_idx"].to_numpy(dtype="int32")
            origin_times = df.iloc[rows]["time"].to_numpy()
            fold = np.full(len(rows), int(val_year), dtype="int16")
            for level in GROUP_LEVELS[group]:
                for hour in HOURS:
                    center = V1.hres_dir_from_rows(df, rows, group, level, horizon, hour)
                    actual = V1.target_dir(cube, origin_times, grid_idx, level, horizon, hour)
                    ok = np.isfinite(center) & np.isfinite(actual)
                    if not bool(ok.any()):
                        continue
                    y_ok = actual[ok]
                    c_ok = center[ok]
                    f_ok = fold[ok]
                    append_block(store, block_key("group", region, group, horizon), y_ok, c_ok, f_ok)
                    append_block(store, block_key("level", region, group, horizon, level=level), y_ok, c_ok, f_ok)
                    append_block(store, block_key("hour", region, group, horizon, hour=hour), y_ok, c_ok, f_ok)
                    append_block(store, block_key("level_hour", region, group, horizon, level=level, hour=hour), y_ok, c_ok, f_ok)

    out = {}
    for key, parts in store.items():
        out[key] = {
            "y": np.concatenate(parts["y"]),
            "center": np.concatenate(parts["center"]),
            "fold": np.concatenate(parts["fold"]),
        }
    return out


def current_half_widths(base_csv: Path) -> dict[tuple[str, str, str, int, str, str], float]:
    usecols = ["type", "region", "horizon", "hour", "level", "dir_05", "dir_95"]
    chunks = []
    for chunk in pd.read_csv(base_csv, usecols=usecols, chunksize=600_000):
        grid = chunk[chunk["type"].eq("grid") & chunk["horizon"].isin(HORIZONS)].copy()
        if grid.empty:
            continue
        grid["group"] = np.where(grid["level"].isin(GROUP_LEVELS["surface"]), "surface", "pressure")
        grid["half_width"] = ((pd.to_numeric(grid["dir_95"], errors="coerce") - pd.to_numeric(grid["dir_05"], errors="coerce")) % 360.0) / 2.0
        chunks.append(grid[["region", "group", "horizon", "hour", "level", "half_width"]])
    if not chunks:
        raise RuntimeError(f"no grid direction rows found in {base_csv}")
    df = pd.concat(chunks, ignore_index=True)
    widths: dict[tuple[str, str, str, int, str, str], float] = {}
    for (region, group, horizon), s in df.groupby(["region", "group", "horizon"])["half_width"]:
        widths[block_key("group", region, group, int(horizon))] = float(s.median())
    for (region, group, horizon, level), s in df.groupby(["region", "group", "horizon", "level"])["half_width"]:
        widths[block_key("level", region, group, int(horizon), level=str(level))] = float(s.median())
    for (region, group, horizon, hour), s in df.groupby(["region", "group", "horizon", "hour"])["half_width"]:
        widths[block_key("hour", region, group, int(horizon), hour=int(hour))] = float(s.median())
    for (region, group, horizon, level, hour), s in df.groupby(["region", "group", "horizon", "level", "hour"])["half_width"]:
        widths[block_key("level_hour", region, group, int(horizon), level=str(level), hour=int(hour))] = float(s.median())
    return widths


def proxy_alignment_blocks(base_csv: Path, region: str, group: str, horizon: int) -> dict[tuple[str, str, str, int, str, str], tuple[float, float]]:
    levels = set(GROUP_LEVELS[group])
    hres = V1.inference_hres_centers(region, group, horizon)
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
        merged["group"] = group
        merged["diff"] = V1.circ_abs_diff(merged["dir_50"].to_numpy(dtype="float64"), merged["center"].to_numpy(dtype="float64"))
        diffs.append(merged[["region", "group", "horizon", "hour", "level", "diff"]])
    if not diffs:
        raise RuntimeError(f"no base rows for {region}/{group}/d{horizon}")
    df = pd.concat(diffs, ignore_index=True)
    out: dict[tuple[str, str, str, int, str, str], tuple[float, float]] = {}

    def add(key, s: pd.Series) -> None:
        out[key] = (float(np.nanmedian(s)), float(np.nanmean(s)))

    for _, s in df.groupby(["region", "group", "horizon"])["diff"]:
        first = df.loc[s.index[0]]
        add(block_key("group", first["region"], first["group"], int(first["horizon"])), s)
    for _, s in df.groupby(["region", "group", "horizon", "level"])["diff"]:
        first = df.loc[s.index[0]]
        add(block_key("level", first["region"], first["group"], int(first["horizon"]), level=str(first["level"])), s)
    for _, s in df.groupby(["region", "group", "horizon", "hour"])["diff"]:
        first = df.loc[s.index[0]]
        add(block_key("hour", first["region"], first["group"], int(first["horizon"]), hour=int(first["hour"])), s)
    for _, s in df.groupby(["region", "group", "horizon", "level", "hour"])["diff"]:
        first = df.loc[s.index[0]]
        add(
            block_key("level_hour", first["region"], first["group"], int(first["horizon"]), level=str(first["level"]), hour=int(first["hour"])),
            s,
        )
    return out


def evaluate_blocks(base_csv: Path, grid_per_anchor: int, seed: int) -> pd.DataFrame:
    current = current_half_widths(base_csv)
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        for group in GROUP_LEVELS:
            for horizon in HORIZONS:
                log(f"[cv] {region}/{group}/d{horizon}")
                blocks = collect_validation_blocks(region, group, horizon, grid_per_anchor, seed)
                proxy = proxy_alignment_blocks(base_csv, region, group, horizon)
                for key, arrs in blocks.items():
                    if len(arrs["y"]) < MIN_N:
                        continue
                    current_hw = current.get(key)
                    if current_hw is None or not np.isfinite(current_hw):
                        continue
                    _, selected_hw = best_width(arrs["y"], arrs["center"])
                    current_scores = []
                    selected_scores = []
                    for val_year in VAL_YEARS:
                        m = arrs["fold"] == int(val_year)
                        current_scores.append(V1.circular_winkler_score(arrs["y"][m], arrs["center"][m], current_hw))
                        selected_scores.append(V1.circular_winkler_score(arrs["y"][m], arrs["center"][m], selected_hw))
                    current_mean = float(np.mean(current_scores))
                    selected_mean = float(np.mean(selected_scores))
                    fold_gains = np.asarray(current_scores) - np.asarray(selected_scores)
                    proxy_median, proxy_mean = proxy.get(key, (float("inf"), float("inf")))
                    mode = key[0]
                    gate_passed = (
                        selected_mean <= current_mean - float(MIN_MEAN_GAIN[mode])
                        and float(np.min(fold_gains)) >= -MAX_ALLOWED_WORSE_FOLD
                        and abs(selected_hw - current_hw) >= MIN_WIDTH_CHANGE
                        and proxy_median <= MAX_CENTER_PROXY_MEDIAN_DIFF
                        and proxy_mean <= MAX_CENTER_PROXY_MEAN_DIFF
                    )
                    row = key_record(key)
                    row.update(
                        {
                            "n": int(len(arrs["y"])),
                            "current_hw": float(current_hw),
                            "selected_hw": float(selected_hw),
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
                    rows.append(row)
    return pd.DataFrame(rows)


def apply_widths(base_csv: Path, summary: pd.DataFrame, output_csv: Path, output_zip: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    selected = summary[summary["gate_passed"].astype(bool)].copy()
    selected = selected.sort_values(["specificity", "cv_mean_gain"], ascending=[False, False], kind="mergesort")
    df = pd.read_csv(base_csv, low_memory=False)
    applied = np.zeros(len(df), dtype=bool)
    counts: dict[str, int] = {}
    for _, row in selected.iterrows():
        region = str(row["region"])
        group = str(row["group"])
        horizon = int(row["horizon"])
        level = str(row["level"])
        hour = str(row["hour"])
        half_width = float(row["selected_hw"])
        levels = set(GROUP_LEVELS[group]) if level == "*" else {level}
        m = (
            df["type"].eq("grid")
            & df["region"].eq(region)
            & df["horizon"].eq(horizon)
            & df["level"].isin(levels)
        )
        if hour != "*":
            m &= df["hour"].astype(int).eq(int(hour))
        idx = df.index[m.to_numpy(dtype=bool) & ~applied]
        if len(idx) == 0:
            continue
        center = pd.to_numeric(df.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        df.loc[idx, "dir_05"] = (center - half_width) % 360.0
        df.loc[idx, "dir_95"] = (center + half_width) % 360.0
        applied[idx.to_numpy(dtype=int)] = True
        name = f"{region}_{group}_d{horizon}_{row['mode']}_{level}_h{hour}_hw{half_width:g}"
        counts[name] = int(len(idx))

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
    parser.add_argument("--grid-per-anchor", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    if not args.base_csv.exists():
        raise SystemExit(f"missing base CSV: {args.base_csv}")
    if len(args.output_zip.name) >= 64:
        raise SystemExit(f"zip filename is too long for Codabench: {args.output_zip.name}")

    log("Direction-width portfolio v2")
    log(f"Base CSV: {args.base_csv} ({args.base_csv.stat().st_size:,} bytes)")
    summary = evaluate_blocks(args.base_csv, args.grid_per_anchor, args.seed)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    log(f"Wrote CV summary: {args.summary_csv}")
    log(summary.sort_values("cv_mean_gain", ascending=False).head(40).to_string(index=False))

    selected = summary[summary["gate_passed"].astype(bool)].copy()
    if selected.empty:
        payload = {
            "status": "gate_failed_no_submission_written",
            "reason": "No fine-grained direction-width block passed the chronological CV and center-proxy gates.",
            "base_csv": str(args.base_csv.resolve()),
            "cv_summary_csv": str(args.summary_csv.resolve()),
            "best_candidates": summary.sort_values("cv_mean_gain", ascending=False).head(40).to_dict(orient="records"),
            "compliance": {
                "external_training_data_used": False,
                "web_data_used": False,
                "evaluation_target_labels_used_for_training": False,
                "official_dataset_root": str(DATA.resolve()),
            },
        }
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("No gated width portfolio v2 candidate was emitted.")

    final, counts = apply_widths(args.base_csv, summary, args.output_csv, args.output_zip)
    with __import__("zipfile").ZipFile(args.output_zip) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")

    payload = {
        "status": "completed",
        "base_csv": {
            "path": str(args.base_csv.resolve()),
            "size": args.base_csv.stat().st_size,
            "sha256": sha256(args.base_csv),
        },
        "cv_summary_csv": str(args.summary_csv.resolve()),
        "selected_blocks": selected.sort_values(["specificity", "cv_mean_gain"], ascending=[False, False]).to_dict(orient="records"),
        "patch_counts": counts,
        "submission": {
            "csv": str(args.output_csv.resolve()),
            "csv_size": args.output_csv.stat().st_size,
            "csv_sha256": sha256(args.output_csv),
            "zip": str(args.output_zip.resolve()),
            "zip_size": args.output_zip.stat().st_size,
            "zip_name_length": len(args.output_zip.name),
            "zip_sha256": sha256(args.output_zip),
            "internal_names": names,
            "internal_csv_size": int(info.file_size),
            "testzip": bad,
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
                "Width gates use official historical HRES features and reanalysis targets.",
                "Inference center-proxy alignment uses official inference HRES features but no evaluation targets.",
                "Final patch changes only dir_05/dir_95 around the generated end-to-end dir_50 centers.",
                "Overlapping selected blocks are applied from most specific to least specific.",
            ],
        },
        "code_hashes": {
            "build_direction_width_portfolio_v2_candidate.py": sha256(Path(__file__).resolve()),
            "build_direction_width_portfolio_v1_candidate.py": sha256(Path("build_direction_width_portfolio_v1_candidate.py")),
            "sea_winds_end_to_end_final.py": sha256(Path("sea_winds_end_to_end_final.py")),
        },
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {args.manifest}")
    log(f"OK: {args.output_zip}")


if __name__ == "__main__":
    main()
