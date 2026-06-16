from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_speedpos_wcal.csv"
OUT_CSV = WORK / "pred_stndir_wexp.csv"
OUT_ZIP = WORK / "sub_stndir_wexp.zip"
MANIFEST = WORK / "manifest_stndir_wexp.json"

COLS = [
    "type",
    "window",
    "region",
    "latitude",
    "longitude",
    "station",
    "horizon",
    "hour",
    "level",
    "q05",
    "q50",
    "q95",
    "dir_05",
    "dir_50",
    "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]

# The preceding public run showed that tightening these station direction
# intervals was harmful.  These blocks therefore only widen existing intervals
# around the unchanged center, targeting the large station-direction gaps.
WIDTH_BLOCKS = [
    {
        "name": "ns_station_d7_widen",
        "region": "north_sea",
        "horizon": 7,
        "target_half_width": 145.0,
        "public_tighten_result": "312.3524 -> 321.8857 when tightened to 102.5",
    },
    {
        "name": "ns_station_d14_widen",
        "region": "north_sea",
        "horizon": 14,
        "target_half_width": 155.0,
        "public_tighten_result": "303.5619 -> 325.1810 when tightened to 115.0",
    },
    {
        "name": "ecs_station_d14_widen",
        "region": "east_china_sea",
        "horizon": 14,
        "target_half_width": 150.0,
        "public_tighten_result": "345.2232 -> 370.2054 when tightened to 107.5",
    },
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
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


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLS].copy()
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def row_diff(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def apply_width_blocks(df: pd.DataFrame) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for block in WIDTH_BLOCKS:
        mask = (
            df["type"].eq("station")
            & df["region"].eq(str(block["region"]))
            & df["horizon"].eq(int(block["horizon"]))
        )
        idx = df.index[mask]
        if len(idx) == 0:
            raise SystemExit(f"Direction width block matched no rows: {block['name']}")
        lo_old = pd.to_numeric(df.loc[idx, "dir_05"], errors="coerce").to_numpy(dtype="float64")
        center = pd.to_numeric(df.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
        hi_old = pd.to_numeric(df.loc[idx, "dir_95"], errors="coerce").to_numpy(dtype="float64")
        old_hw = ((hi_old - lo_old) % 360.0) / 2.0
        target_hw = float(block["target_half_width"])
        new_hw = np.maximum(old_hw, target_hw)
        new_hw = np.minimum(new_hw, 179.9)
        df.loc[idx, "dir_05"] = (center - new_hw) % 360.0
        df.loc[idx, "dir_95"] = (center + new_hw) % 360.0
        changed = np.abs(new_hw - old_hw) > 1e-9
        reports.append(
            {
                "name": block["name"],
                "region": block["region"],
                "horizon": int(block["horizon"]),
                "target_rows": int(len(idx)),
                "changed_rows": int(changed.sum()),
                "old_half_width_mean": float(np.nanmean(old_hw)),
                "new_half_width_mean": float(np.nanmean(new_hw)),
                "target_half_width": target_hw,
                "public_tighten_result": block["public_tighten_result"],
            }
        )
    return reports


def validate(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    for c in SPEED_COLS:
        after[c] = pd.to_numeric(after[c], errors="coerce").clip(lower=0).round(2)
    after["q05"] = after[["q05", "q50"]].min(axis=1).round(2)
    after["q95"] = after[["q95", "q50"]].max(axis=1).round(2)
    for c in DIR_COLS:
        after[c] = ((pd.to_numeric(after[c], errors="coerce") % 360.0).round(1) % 360.0).round(1)

    grid = after["type"].eq("grid")
    type_counts = after["type"].value_counts(dropna=False).to_dict()
    missing_pred = int(after[SPEED_COLS + DIR_COLS].isna().any(axis=1).sum())
    missing_grid_key = int(after.loc[grid, ["window", "region", "latitude", "longitude", "horizon", "hour", "level"]].isna().any(axis=1).sum())
    missing_station_key = int(after.loc[~grid, ["window", "region", "station", "horizon", "hour"]].isna().any(axis=1).sum())
    bad_speed = int(((after["q05"] > after["q50"]) | (after["q50"] > after["q95"]) | (after[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((after[DIR_COLS] < 0) | (after[DIR_COLS] >= 360) | after[DIR_COLS].isna()).any(axis=1).sum())
    grid_dup = int(after.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(after.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    speed_changed = row_diff(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = row_diff(before, after, DIR_COLS, 1, circular=True)
    center_speed_changed = row_diff(before, after, ["q50"], 2, circular=False)
    center_dir_changed = row_diff(before, after, ["dir_50"], 1, circular=True)

    if len(after) != 3_448_800 or type_counts.get("grid") != 3_447_360 or type_counts.get("station") != 1_440:
        raise SystemExit(f"row/type count validation failed: rows={len(after)} counts={type_counts}")
    if missing_pred or missing_grid_key or missing_station_key or bad_speed or bad_dir or grid_dup or station_dup:
        raise SystemExit(
            f"content validation failed: missing_pred={missing_pred} "
            f"missing_grid_key={missing_grid_key} missing_station_key={missing_station_key} "
            f"bad_speed={bad_speed} bad_dir={bad_dir} grid_dup={grid_dup} station_dup={station_dup}"
        )
    if int(center_speed_changed.sum()) or int(center_dir_changed.sum()):
        raise SystemExit(
            f"center changed unexpectedly: speed_centers={int(center_speed_changed.sum())} "
            f"dir_centers={int(center_dir_changed.sum())}"
        )
    # This branch must not touch speed intervals. It is layered on top of the
    # already-generated speed-positive base.
    if int(speed_changed.sum()):
        raise SystemExit(f"unexpected speed deltas inside station direction branch: {int(speed_changed.sum())}")

    return {
        "rows": int(len(after)),
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "speed_interval_rows_changed": int(speed_changed.sum()),
        "direction_interval_rows_changed": int(dir_changed.sum()),
        "speed_center_rows_changed": int(center_speed_changed.sum()),
        "direction_center_rows_changed": int(center_dir_changed.sum()),
        "missing_prediction_rows": missing_pred,
        "missing_grid_key_rows": missing_grid_key,
        "missing_station_key_rows": missing_station_key,
        "bad_speed_rows": bad_speed,
        "bad_direction_rows": bad_dir,
        "grid_duplicate_keys": grid_dup,
        "station_duplicate_keys": station_dup,
    }


def write_outputs(df: pd.DataFrame) -> None:
    print(f"Writing {OUT_CSV}", flush=True)
    df[COLS].to_csv(OUT_CSV, index=False)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    if len(OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {OUT_ZIP.name}")
    print(f"zip={OUT_ZIP} size={OUT_ZIP.stat().st_size:,} uncompressed={info.file_size:,}", flush=True)


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing speed-positive base CSV: {BASE_CSV}. Run .\\run_speedpos_width_cal_e2e.ps1 first.")
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    before = normalize(pd.read_csv(BASE_CSV, low_memory=False))
    after = before.copy()
    reports = apply_width_blocks(after)
    audit = validate(before, after)
    write_outputs(after)
    manifest = {
        "status": "submission_written",
        "out_csv": str(OUT_CSV),
        "out_zip": str(OUT_ZIP),
        "zip_predictions_sha256": sha256_zip_member(OUT_ZIP),
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": sha256(BASE_CSV),
        "audit": audit,
        "station_direction_width_blocks": reports,
        "public_feedback_basis": {
            "source_submission": "sub_biggap_wcal.zip",
            "observation": "Tightening station direction intervals worsened all three targeted large station-direction dimensions, so this candidate tests widening only while preserving centers.",
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "future_target_data_used": False,
            "notes": [
                "All changes are direction interval widening around centers generated by the official-data pipeline.",
                "No q50 or dir_50 center is changed.",
                "The builder does not read evaluation labels or any external dataset.",
            ],
        },
        "code_hashes": {
            "build_station_dir_width_expand_candidate.py": sha256(Path(__file__).resolve()),
            "run_station_dir_width_expand_e2e.ps1": sha256(ROOT / "run_station_dir_width_expand_e2e.ps1"),
            "run_speedpos_width_cal_e2e.ps1": sha256(ROOT / "run_speedpos_width_cal_e2e.ps1"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST}", flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
