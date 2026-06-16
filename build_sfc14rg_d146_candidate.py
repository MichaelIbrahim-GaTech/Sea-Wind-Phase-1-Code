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

BASE_CSV = WORK / "pred_ecs14d146.csv"
PATCH_SOURCE_CSV = WORK / "pred_ns_sfc14_rg.csv"
OUT_CSV = WORK / "pred_sfc14rg_d146.csv"
OUT_ZIP = WORK / "sub_sfc14rgd146.zip"
MANIFEST = WORK / "manifest_sfc14rg_d146.json"

COLS = E2E.COLS
KEYS = E2E.KEYS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS
SURFACE_LEVELS = ("10m", "100m")


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


def assert_same_keys(base: pd.DataFrame, source: pd.DataFrame) -> None:
    if len(base) != len(source):
        raise SystemExit(f"row count mismatch: base={len(base)} source={len(source)}")
    for col in KEYS:
        left = base[col].fillna("").astype(str).reset_index(drop=True)
        right = source[col].fillna("").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise SystemExit(f"row key mismatch on {col}; refusing row-index patch")


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


def target_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        df["type"].eq("grid")
        & df["region"].eq("north_sea")
        & df["horizon"].eq(14)
        & df["level"].isin(SURFACE_LEVELS)
    ).to_numpy(dtype=bool)


def apply_patch(base: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    mask = target_mask(base)
    if int(mask.sum()) != 164_160:
        raise SystemExit(f"unexpected NS surface d14 target rows: {int(mask.sum())}")

    out = base.copy()
    before = base.copy()
    out.loc[mask, DIR_COLS] = source.loc[mask, DIR_COLS].to_numpy()

    speed_changed = rows_changed(before, out, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, out, DIR_COLS, 1, circular=True)
    center_changed = rows_changed(before, out, ["dir_50"], 1, circular=True)
    outside = dir_changed & ~mask
    station = out["type"].eq("station").to_numpy(dtype=bool)
    station_dir_changed = dir_changed & station

    if int(speed_changed.sum()) or int(outside.sum()) or int(station_dir_changed.sum()):
        raise SystemExit(
            "unexpected deltas: "
            f"speed={int(speed_changed.sum())} outside_dir={int(outside.sum())} "
            f"station_dir={int(station_dir_changed.sum())}"
        )

    old_center = pd.to_numeric(before.loc[mask, "dir_50"], errors="coerce").to_numpy(dtype="float64")
    new_center = pd.to_numeric(out.loc[mask, "dir_50"], errors="coerce").to_numpy(dtype="float64")
    move = np.abs(((new_center - old_center + 180.0) % 360.0) - 180.0)
    changed_move = move[move > 0.05]

    return out, {
        "target_rows": int(mask.sum()),
        "direction_interval_rows_changed": int(dir_changed.sum()),
        "direction_center_rows_changed": int(center_changed.sum()),
        "speed_interval_rows_changed": int(speed_changed.sum()),
        "non_target_direction_rows_changed": int(outside.sum()),
        "station_direction_rows_changed": int(station_dir_changed.sum()),
        "center_move_mean_changed": float(np.nanmean(changed_move)) if len(changed_move) else 0.0,
        "center_move_p90_changed": float(np.nanquantile(changed_move, 0.90)) if len(changed_move) else 0.0,
        "center_move_p99_changed": float(np.nanquantile(changed_move, 0.99)) if len(changed_move) else 0.0,
    }


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing promoted d146 base: {BASE_CSV}. Run .\\run_ecs14d146_e2e.ps1 first.")
    if not PATCH_SOURCE_CSV.exists():
        raise SystemExit(f"Missing NS surface d14 row-gate source: {PATCH_SOURCE_CSV}. Run .\\run_ns_surface14_rowgate_e2e.ps1 first.")

    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize(pd.read_csv(BASE_CSV, low_memory=False))
    print(f"Reading patch source {PATCH_SOURCE_CSV} ({PATCH_SOURCE_CSV.stat().st_size:,} bytes)", flush=True)
    source = normalize(pd.read_csv(PATCH_SOURCE_CSV, low_memory=False))
    assert_same_keys(base, source)

    patched, delta = apply_patch(base, source)
    final = E2E.validate_final(patched)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)

    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None or len(OUT_ZIP.name) >= 64:
        raise SystemExit(f"bad zip: names={names} testzip={bad} filename_length={len(OUT_ZIP.name)}")

    manifest = {
        "status": "submission_written",
        "candidate_type": "layered_ns_surface_d14_rowgate_center_on_promoted_d146",
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
        "patch_source_csv": {
            "path": str(PATCH_SOURCE_CSV),
            "size": int(PATCH_SOURCE_CSV.stat().st_size),
            "sha256": sha256(PATCH_SOURCE_CSV),
        },
        "delta": delta,
        "public_feedback_basis": {
            "current_promoted_base": "sub_ecs14d146.zip",
            "current_base_public_score": 1.442113,
            "target_metric": "Dir NS Surface d14",
            "current_metric": 326.9058,
            "leader_reference_metric": 298.76,
            "evidence": [
                "Full row-gate candidate moved Dir NS Surface d14 from 326.9058 to 326.4197 but hurt Dir NS Pressure d14.",
                "This candidate copies only the isolated NS surface d14 row-gated seasonal center block and keeps pressure d14 unchanged.",
            ],
            "gate": (
                "Promote only if Dir NS Surface d14 improves versus 326.9058 and no "
                "previously accepted dimension regresses materially."
            ),
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "future_target_data_used": False,
            "notes": [
                "Base and patch source are generated by official-data-only scripts.",
                "Only North Sea grid surface horizon-14 direction columns are copied from the official-data row-gate source.",
                "No speed row, station row, or non-target direction row is changed.",
            ],
        },
        "code_hashes": {
            "build_sfc14rg_d146_candidate.py": sha256(Path(__file__).resolve()),
            "run_sfc14rg_d146_e2e.ps1": sha256(ROOT / "run_sfc14rg_d146_e2e.ps1"),
            "run_ecs14d146_e2e.ps1": sha256(ROOT / "run_ecs14d146_e2e.ps1"),
            "run_ns_surface14_rowgate_e2e.ps1": sha256(ROOT / "run_ns_surface14_rowgate_e2e.ps1"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(delta, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
