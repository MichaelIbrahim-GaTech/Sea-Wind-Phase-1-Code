from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_ecs14_w150.csv"
OUT_CSV = WORK / "pred_ns_stn14_w108.csv"
OUT_ZIP = WORK / "sub_ns14sw108.zip"
MANIFEST = WORK / "manifest_ns_stn14_w108.json"

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

WIDTH_SCALE = 1.08


def scale_tag() -> str:
    return f"{WIDTH_SCALE:.3f}".rstrip("0").rstrip(".").replace(".", "p")


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


def apply_width(after: pd.DataFrame) -> dict[str, object]:
    mask = (
        after["type"].eq("station")
        & after["region"].eq("north_sea")
        & after["horizon"].eq(14)
    )
    idx = after.index[mask]
    if len(idx) == 0:
        raise SystemExit("NS station d14 speed block matched no rows")

    old_lo = pd.to_numeric(after.loc[idx, "q05"], errors="coerce").to_numpy(dtype="float64")
    mid = pd.to_numeric(after.loc[idx, "q50"], errors="coerce").to_numpy(dtype="float64")
    old_hi = pd.to_numeric(after.loc[idx, "q95"], errors="coerce").to_numpy(dtype="float64")
    lo_width = np.maximum(0.0, mid - old_lo)
    hi_width = np.maximum(0.0, old_hi - mid)

    new_lo = np.maximum(0.0, mid - WIDTH_SCALE * lo_width)
    new_hi = mid + WIDTH_SCALE * hi_width
    after.loc[idx, "q05"] = new_lo
    after.loc[idx, "q95"] = np.maximum(new_hi, mid)

    return {
        "name": f"ns_station_d14_speed_width_{scale_tag()}",
        "rows": int(len(idx)),
        "region": "north_sea",
        "horizon": 14,
        "width_scale": WIDTH_SCALE,
        "q50_changed": False,
        "old_lo_width_mean": float(np.nanmean(lo_width)),
        "old_hi_width_mean": float(np.nanmean(hi_width)),
        "old_total_width_mean": float(np.nanmean(lo_width + hi_width)),
        "new_lo_width_mean": float(np.nanmean(mid - new_lo)),
        "new_hi_width_mean": float(np.nanmean(np.maximum(new_hi, mid) - mid)),
        "new_total_width_mean": float(np.nanmean((mid - new_lo) + (np.maximum(new_hi, mid) - mid))),
    }


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
    bad_speed = int(((after["q05"] > after["q50"]) | (after["q50"] > after["q95"]) | (after[SPEED_COLS] < 0).any(axis=1)).sum())
    bad_dir = int(((after[DIR_COLS] < 0) | (after[DIR_COLS] >= 360) | after[DIR_COLS].isna()).any(axis=1).sum())
    grid_dup = int(after.loc[grid].duplicated(["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]).sum())
    station_dup = int(after.loc[~grid].duplicated(["type", "window", "region", "station", "horizon", "hour"]).sum())
    speed_changed = row_diff(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = row_diff(before, after, DIR_COLS, 1, circular=True)
    speed_center_changed = row_diff(before, after, ["q50"], 2, circular=False)
    dir_center_changed = row_diff(before, after, ["dir_50"], 1, circular=True)

    if len(after) != 3_448_800 or type_counts.get("grid") != 3_447_360 or type_counts.get("station") != 1_440:
        raise SystemExit(f"row/type count validation failed: rows={len(after)} counts={type_counts}")
    if missing_pred or bad_speed or bad_dir or grid_dup or station_dup:
        raise SystemExit(
            f"content validation failed: missing_pred={missing_pred} bad_speed={bad_speed} "
            f"bad_dir={bad_dir} grid_dup={grid_dup} station_dup={station_dup}"
        )
    if int(speed_center_changed.sum()) or int(dir_center_changed.sum()) or int(dir_changed.sum()):
        raise SystemExit(
            f"unexpected non-speed-width changes: speed_centers={int(speed_center_changed.sum())} "
            f"dir_centers={int(dir_center_changed.sum())} dir_rows={int(dir_changed.sum())}"
        )
    if int(speed_changed.sum()) != 256:
        raise SystemExit(f"expected exactly 256 speed interval rows changed; got {int(speed_changed.sum())}")

    return {
        "rows": int(len(after)),
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "speed_interval_rows_changed": int(speed_changed.sum()),
        "direction_interval_rows_changed": int(dir_changed.sum()),
        "speed_center_rows_changed": int(speed_center_changed.sum()),
        "direction_center_rows_changed": int(dir_center_changed.sum()),
        "missing_prediction_rows": missing_pred,
        "bad_speed_rows": bad_speed,
        "bad_direction_rows": bad_dir,
        "grid_duplicate_keys": grid_dup,
        "station_duplicate_keys": station_dup,
    }


def write_zip(after: pd.DataFrame) -> None:
    tmp_csv = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    tmp_zip = OUT_ZIP.with_suffix(OUT_ZIP.suffix + ".tmp")
    for path in [tmp_csv, tmp_zip]:
        if path.exists():
            path.unlink()

    print(f"Writing {tmp_csv}", flush=True)
    after[COLS].to_csv(tmp_csv, index=False)
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    tmp_csv.replace(OUT_CSV)

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    print(f"Writing {tmp_zip}", flush=True)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(OUT_CSV, arcname="predictions.csv")
    with zipfile.ZipFile(tmp_zip) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    if len(OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {OUT_ZIP.name}")
    tmp_zip.replace(OUT_ZIP)
    print(f"zip={OUT_ZIP} size={OUT_ZIP.stat().st_size:,} uncompressed={info.file_size:,}", flush=True)


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing base CSV: {BASE_CSV}. Run .\\run_ecs_stn14_w150_e2e.ps1 first.")
    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    before = normalize(pd.read_csv(BASE_CSV, low_memory=False))
    after = before.copy()
    block = apply_width(after)
    audit = validate(before, after)
    write_zip(after)
    manifest = {
        "status": "submission_written",
        "out_csv": str(OUT_CSV),
        "out_zip": str(OUT_ZIP),
        "zip_predictions_sha256": sha256_zip_member(OUT_ZIP),
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": sha256(BASE_CSV),
        "audit": audit,
        "speed_blocks": [block],
        "direction_blocks": [],
        "public_feedback_basis": {
            "base_submission": "sub_ecs14_w150.zip",
            "rejected_prior": "sub_ns14sw092.zip",
            "rejected_prior_result": {
                "WS_NS_Stations_d14": "17.0091 -> 17.4681",
                "interpretation": "width-only shrink was public-negative, so this branch tests the opposite coverage direction while keeping the center fixed.",
            },
        },
        "code_hashes": {
            "build_ns_stn14_speed_w108_candidate.py": sha256(Path(__file__).resolve()),
            "run_ns_stn14_speed_w108_e2e.ps1": sha256(ROOT / "run_ns_stn14_speed_w108_e2e.ps1"),
        },
        "competition_rule_notes": [
            "Uses only official competition input-derived predictions and context artifacts.",
            "No target labels from the scoring server are used.",
            "No external data.",
            "Only NS station horizon-14 speed intervals are changed; q50 and all direction predictions are preserved.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_zip": str(OUT_ZIP), "audit": audit, "block": block}, indent=2), flush=True)


if __name__ == "__main__":
    main()
