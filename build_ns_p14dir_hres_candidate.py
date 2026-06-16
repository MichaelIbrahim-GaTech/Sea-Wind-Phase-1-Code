from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E
from build_rankaware_pressure_d14_hresd10_v1_candidate import HALF_WIDTH, load_hres_d10_pressure_ns


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_stndir_d1_bag_gate.csv"
OUT_CSV = WORK / "pred_ns_p14dir_hres.csv"
OUT_ZIP = WORK / "sub_ns_p14dir_hres.zip"
MANIFEST = WORK / "manifest_ns_p14dir_hres.json"

REGION = "north_sea"
HORIZON = 14
LEVELS = E2E.PRESSURE_LEVELS
COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS


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


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    return out


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if set(cols) == set(DIR_COLS):
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def apply_patch(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("Loading official North Sea HRES d10 pressure-vector directions", flush=True)
    hres = load_hres_d10_pressure_ns()
    mask = (
        df["type"].eq("grid")
        & df["region"].eq(REGION)
        & df["horizon"].eq(HORIZON)
        & df["level"].isin(LEVELS)
    )
    lookup = df.loc[mask].reset_index()[["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]]
    if len(lookup) != 410_400:
        raise SystemExit(f"unexpected NS pressure d14 target rows: {len(lookup):,}")
    merged = lookup.merge(
        hres,
        on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
        how="left",
        validate="one_to_one",
    )
    missing = int(merged["hres_dir"].isna().sum())
    if missing:
        raise SystemExit(f"missing HRES d10 pressure directions: {missing}")
    idx = merged["index"].to_numpy(dtype="int64")
    center = merged["hres_dir"].to_numpy(dtype="float64") % 360.0
    out = df.copy()
    out.loc[idx, "dir_50"] = center
    out.loc[idx, "dir_05"] = (center - float(HALF_WIDTH)) % 360.0
    out.loc[idx, "dir_95"] = (center + float(HALF_WIDTH)) % 360.0
    print(f"Patched NS pressure d14 direction rows: {len(idx):,}; half_width={HALF_WIDTH}", flush=True)
    return out, mask


def validate_delta(before: pd.DataFrame, after: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    speed_changed = rows_changed(before, after, SPEED_COLS, 2)
    dir_changed = rows_changed(before, after, DIR_COLS, 1)
    target_arr = target.to_numpy(dtype=bool)
    outside = dir_changed & ~target_arr
    if int(speed_changed.sum()) != 0:
        raise SystemExit(f"unexpected speed deltas: {int(speed_changed.sum())}")
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected direction deltas outside NS pressure d14: {int(outside.sum())}")
    return {
        "target_rows": int(target_arr.sum()),
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_direction_rows_changed": int(outside.sum()),
        "target_rows_unchanged_after_rounding": int((target_arr & ~dir_changed).sum()),
    }


def write_manifest(final: pd.DataFrame, delta: dict[str, object]) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")
    manifest = {
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
        "delta": delta,
        "component": {
            "target": "Dir NS Pressure d14",
            "region": REGION,
            "levels": list(LEVELS),
            "horizon": HORIZON,
            "hres_lead_used": 10,
            "half_width": float(HALF_WIDTH),
            "training_backtest_summary": str(WORK / "robust_hres_direction_backtest_summary.csv"),
            "training_backtest_mean_score": 333.99884030914274,
            "training_backtest_max_score": 336.1596599570423,
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
            "bad_speed": 0,
            "bad_dir": 0,
            "missing": 0,
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "notes": [
                "Uses only official provided inference HRES pressure-vector fields for the prediction windows.",
                "Uses official training reanalysis only for the robust_hres_direction_backtest summary.",
                "Final 2022 inference does not read public/evaluation targets.",
                "No external datasets and no missing-data imputation are used.",
            ],
        },
        "code_hashes": {
            "build_ns_p14dir_hres_candidate.py": sha256(Path(__file__).resolve()),
            "build_rankaware_pressure_d14_hresd10_v1_candidate.py": sha256(ROOT / "build_rankaware_pressure_d14_hresd10_v1_candidate.py"),
            "robust_hres_direction_backtest.py": sha256(ROOT / "robust_hres_direction_backtest.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_station_d1_direction_bag_gate_e2e.ps1 first.")
    print(f"Reading current best base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    before = base.copy()
    patched, target = apply_patch(base)
    final = E2E.validate_final(patched)
    delta = validate_delta(before, final, target)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, delta)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
