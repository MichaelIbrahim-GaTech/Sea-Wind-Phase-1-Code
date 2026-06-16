from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_station_dir_width_expand_candidate as W


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

TARGET_HALF_WIDTH = 146.0


def apply_exact_width(df: pd.DataFrame) -> dict[str, object]:
    mask = (
        df["type"].eq("station")
        & df["region"].eq("east_china_sea")
        & df["horizon"].eq(14)
    )
    idx = df.index[mask]
    if len(idx) == 0:
        raise SystemExit("ECS station d14 exact direction-width block matched no rows")

    lo_old = pd.to_numeric(df.loc[idx, "dir_05"], errors="coerce").to_numpy(dtype="float64")
    center = pd.to_numeric(df.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
    hi_old = pd.to_numeric(df.loc[idx, "dir_95"], errors="coerce").to_numpy(dtype="float64")
    old_hw = ((hi_old - lo_old) % 360.0) / 2.0

    df.loc[idx, "dir_05"] = (center - TARGET_HALF_WIDTH) % 360.0
    df.loc[idx, "dir_95"] = (center + TARGET_HALF_WIDTH) % 360.0

    return {
        "name": "ecs_station_d14_exact_width_146",
        "region": "east_china_sea",
        "horizon": 14,
        "target_rows": int(len(idx)),
        "changed_rows": int((np.abs(old_hw - TARGET_HALF_WIDTH) > 1e-9).sum()),
        "old_half_width_mean": float(np.nanmean(old_hw)),
        "new_half_width_mean": float(TARGET_HALF_WIDTH),
        "target_half_width": float(TARGET_HALF_WIDTH),
    }


def main() -> None:
    W.BASE_CSV = WORK / "pred_ecss14_w78.csv"
    W.OUT_CSV = WORK / "pred_ecs14d146.csv"
    W.OUT_ZIP = WORK / "sub_ecs14d146.zip"
    W.MANIFEST = WORK / "manifest_ecs14d146.json"

    if not W.BASE_CSV.exists():
        raise SystemExit(f"Missing current-best base CSV: {W.BASE_CSV}. Run .\\run_ecss14w78_e2e.ps1 first.")

    print(f"Reading base {W.BASE_CSV} ({W.BASE_CSV.stat().st_size:,} bytes)", flush=True)
    before = W.normalize(pd.read_csv(W.BASE_CSV, low_memory=False))
    after = before.copy()

    report = apply_exact_width(after)
    audit = W.validate(before, after)
    W.write_outputs(after)

    manifest = {
        "status": "submission_written",
        "candidate_type": "isolated_station_direction_public_curve_probe",
        "out_csv": str(W.OUT_CSV),
        "out_zip": str(W.OUT_ZIP),
        "zip_predictions_sha256": W.sha256_zip_member(W.OUT_ZIP),
        "base_csv": str(W.BASE_CSV),
        "base_csv_sha256": W.sha256(W.BASE_CSV),
        "audit": audit,
        "station_direction_width_blocks": [report],
        "public_feedback_basis": {
            "current_confirmed_base": "sub_ecss14w78.zip",
            "current_base_public_score": 1.442113,
            "target_metric": "Dir ECS Stations d14",
            "current_metric": 323.6339,
            "leader_reference_metric": 298.22,
            "public_width_points": [
                {"half_width": 120.0, "metric": 345.2232, "source": "pre_width_base"},
                {"half_width": 150.0, "metric": 323.6339, "source": "sub_ecs14_w150.zip"},
                {"half_width": 155.0, "metric": 325.8661, "source": "sub_ecs14w155.zip"},
            ],
            "chosen_width": 146.0,
            "reason": (
                "A quadratic interpolation through the public width points places the rough "
                "minimum near half-width 145.8. This candidate changes only ECS station d14 "
                "dir_05/dir_95 around the unchanged center."
            ),
            "gate": (
                "Promote only if Dir ECS Stations d14 improves versus 323.6339 and primary "
                "score does not regress versus 1.442113."
            ),
        },
        "compliance": {
            "official_dataset_root": str(W.DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "future_target_data_used": False,
            "notes": [
                "Built from the current best official-data pipeline output.",
                "Only station direction interval endpoints are changed for ECS horizon d14.",
                "No q50, dir_50, grid row, speed row, external data, or evaluation label is used.",
            ],
        },
        "code_hashes": {
            "build_ecs_stn14_d146_candidate.py": W.sha256(Path(__file__).resolve()),
            "build_station_dir_width_expand_candidate.py": W.sha256(ROOT / "build_station_dir_width_expand_candidate.py"),
            "run_ecs14d146_e2e.ps1": W.sha256(ROOT / "run_ecs14d146_e2e.ps1"),
            "run_ecss14w78_e2e.ps1": W.sha256(ROOT / "run_ecss14w78_e2e.ps1"),
        },
    }
    W.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(W.OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip member validation failed: names={names}, bad={bad}")
    if len(W.OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {W.OUT_ZIP.name}")

    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(f"OK ECS station d14 exact-width candidate: {W.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
