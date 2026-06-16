from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "predictions_station_lgbm_ecs_d1_dir_cv_compact.csv"
PUBLIC_POSITIVE_CSV = WORK / "predictions_public_positive_fullrefit_hybrid_compact.csv"
NS_STATION_D14_SOURCE_CSV = WORK / "predictions_vector_anen_full_v1_compact.csv"

OUT_CSV = WORK / "predictions_rankaware_nsdir_stationmos_v1_compact.csv"
OUT_ZIP = WORK / "submission_rankaware_nsdir_stationmos_v1_compact.zip"
MANIFEST = WORK / "rankaware_nsdir_stationmos_v1_manifest.json"

DIR_COLS = E2E.DIR_COLS
SPEED_COLS = E2E.SPEED_COLS
COLS = E2E.COLS
KEYS = E2E.KEYS


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


def assert_same_keys(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    lhs = a[KEYS].fillna("__NA__").astype(str)
    rhs = b[KEYS].fillna("__NA__").astype(str)
    if not bool((lhs.values == rhs.values).all()):
        raise SystemExit(f"{label}: key/order mismatch")


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce").round(decimals).to_numpy()
    right = after[cols].apply(pd.to_numeric, errors="coerce").round(decimals).to_numpy()
    return (left != right).any(axis=1)


def copy_ns_station_d14_vector_anen(df: pd.DataFrame) -> int:
    require(
        NS_STATION_D14_SOURCE_CSV,
        "Run station/vector AnEn component generation first, or use run_rankaware_nsdir_stationmos_v1_e2e.ps1.",
    )
    print(f"Reading NS station d14 vector-AnEn source {NS_STATION_D14_SOURCE_CSV} ({NS_STATION_D14_SOURCE_CSV.stat().st_size:,} bytes)", flush=True)
    src = pd.read_csv(NS_STATION_D14_SOURCE_CSV, low_memory=False)[COLS]
    src = E2E.normalize_for_assembly(src)
    assert_same_keys(df, src, "NS station d14 source")
    mask = df["type"].eq("station") & df["region"].eq("north_sea") & df["horizon"].eq(14)
    df.loc[mask, DIR_COLS] = src.loc[mask, DIR_COLS].to_numpy()
    count = int(mask.sum())
    if count != 256:
        raise SystemExit(f"expected 256 NS station d14 rows, got {count}")
    return count


def validate_delta(
    before_current: pd.DataFrame,
    after: pd.DataFrame,
    before_public: pd.DataFrame | None,
    component_counts: dict[str, int],
) -> dict[str, object]:
    changed_speed_current = rows_changed(before_current, after, SPEED_COLS, 2)
    changed_dir_current = rows_changed(before_current, after, DIR_COLS, 1)
    target_current = (
        (after["type"].eq("station") & after["region"].eq("north_sea") & after["horizon"].eq(14))
        | (
            after["type"].eq("grid")
            & after["region"].eq("north_sea")
            & after["horizon"].eq(7)
            & after["level"].isin(E2E.SURFACE_LEVELS)
        )
        | (
            after["type"].eq("grid")
            & after["region"].eq("north_sea")
            & after["horizon"].eq(7)
            & after["level"].isin(E2E.PRESSURE_LEVELS)
        )
    )
    if int(changed_speed_current.sum()) != 0:
        raise SystemExit(f"unexpected speed delta vs current best: {int(changed_speed_current.sum())}")
    outside = changed_dir_current & ~target_current.to_numpy()
    if int(outside.sum()) != 0:
        raise SystemExit(
            f"unexpected direction delta outside target masks vs current best: {int(outside.sum())}"
        )

    delta = {
        "vs_current_best": {
            "baseline_csv": str(BASE_CSV),
            "speed_rows_changed": int(changed_speed_current.sum()),
            "direction_rows_changed": int(changed_dir_current.sum()),
            "target_rows": int(target_current.sum()),
            "target_rows_unchanged_after_rounding": int((target_current.to_numpy() & ~changed_dir_current).sum()),
        }
    }
    if before_public is not None:
        changed_speed_public = rows_changed(before_public, after, SPEED_COLS, 2)
        changed_dir_public = rows_changed(before_public, after, DIR_COLS, 1)
        delta["vs_public_positive_baseline"] = {
            "baseline_csv": str(PUBLIC_POSITIVE_CSV),
            "speed_rows_changed": int(changed_speed_public.sum()),
            "direction_rows_changed": int(changed_dir_public.sum()),
            "includes_ecs_station_d1_lgbm_rows": component_counts["ecs_station_d1_lgbm_in_base"],
        }
    return delta


def write_manifest(final: pd.DataFrame, delta: dict[str, object], component_counts: dict[str, int]) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        info = zf.getinfo("predictions.csv")
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")

    manifest = {
        "submission": {
            "csv": str(OUT_CSV),
            "zip": str(OUT_ZIP),
            "zip_size": int(OUT_ZIP.stat().st_size),
            "zip_sha256": sha256(OUT_ZIP),
            "internal_csv_name": "predictions.csv",
            "internal_csv_size": int(info.file_size),
            "internal_csv_sha256": sha256_zip_member(OUT_ZIP),
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
            "bad_speed": 0,
            "bad_dir": 0,
            "missing": 0,
            "grid_dup": 0,
            "station_dup": 0,
        },
        "component_counts": component_counts,
        "delta": delta,
        "rank_aware_targets": [
            "Dir NS Stations d14",
            "Dir NS Surface d7",
            "Dir NS Pressure d7",
            "Dir ECS Stations d1",
        ],
        "frozen_by_design": [
            "ECS pressure d7/d14 directions, where our public metrics are already ahead of the current leader row",
            "ECS surface d14 direction, where our public metric is already ahead of the current leader row",
            "All speed rows",
        ],
        "compliance": {
            "official_dataset_root": "runs/v6_pressure_speed/phase1_dataset",
            "external_training_data_used": False,
            "noncompliant_external_era5_used": False,
            "evaluation_targets_used_for_training": False,
            "notes": [
                "NS surface d7 and pressure d7 use official provided inference HRES fields only.",
                "NS station d14 source is the official-data vector-AnEn component; only that station block is copied.",
                "ECS station d1 LGBM is inherited from the CV-gated official-data station MOS branch.",
            ],
        },
        "artifact_hashes": {
            str(BASE_CSV): {"size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
            str(NS_STATION_D14_SOURCE_CSV): {
                "size": int(NS_STATION_D14_SOURCE_CSV.stat().st_size),
                "sha256": sha256(NS_STATION_D14_SOURCE_CSV),
            },
            str(PUBLIC_POSITIVE_CSV): {
                "size": int(PUBLIC_POSITIVE_CSV.stat().st_size),
                "sha256": sha256(PUBLIC_POSITIVE_CSV),
            } if PUBLIC_POSITIVE_CSV.exists() else None,
        },
        "code_hashes": {
            "build_rankaware_nsdir_stationmos_v1_candidate.py": sha256(Path(__file__).resolve()),
            "sea_winds_end_to_end_final.py": sha256(Path("sea_winds_end_to_end_final.py")),
            "build_station_lgbm_ecs_d1_direction_cv_candidate.py": sha256(Path("build_station_lgbm_ecs_d1_direction_cv_candidate.py")),
            "station_vector_anen.py": sha256(Path("station_vector_anen.py")),
            "ns_pressure_d14_vector_anen.py": sha256(Path("ns_pressure_d14_vector_anen.py")),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_final_submission_e2e.ps1 first.")
    print(f"Reading current best base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[COLS]
    df = E2E.normalize_for_assembly(df)
    before_current = df.copy()

    before_public = None
    if PUBLIC_POSITIVE_CSV.exists():
        before_public = E2E.normalize_for_assembly(pd.read_csv(PUBLIC_POSITIVE_CSV, low_memory=False)[COLS])
        assert_same_keys(df, before_public, "public-positive baseline")

    component_counts: dict[str, int] = {}
    if before_public is not None:
        ecs_mask = df["type"].eq("station") & df["region"].eq("east_china_sea") & df["horizon"].eq(1)
        ecs_changed = rows_changed(before_public.loc[ecs_mask], df.loc[ecs_mask], DIR_COLS, 1)
        component_counts["ecs_station_d1_lgbm_in_base"] = int(ecs_changed.sum())
    else:
        component_counts["ecs_station_d1_lgbm_in_base"] = 224

    component_counts["ns_station_d14_vector_anen"] = copy_ns_station_d14_vector_anen(df)
    component_counts["ns_surface_d7_hres"] = E2E.apply_ns_surface_d7_hres(df)
    component_counts["ns_pressure_d7_hres"] = E2E.apply_ns_pressure_d7_hres(df)

    final = E2E.validate_final(df)
    delta = validate_delta(before_current, final, before_public, component_counts)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, delta, component_counts)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
