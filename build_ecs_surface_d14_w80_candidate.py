from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_biggap_width_cal_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    B.BASE_CSV = WORK / "pred_ecsp14_s70.csv"
    B.OUT_CSV = WORK / "pred_ecss14_w80.csv"
    B.OUT_ZIP = WORK / "sub_ecss14w80.zip"
    B.MANIFEST = WORK / "manifest_ecss14_w80.json"
    B.SPEED_BLOCKS = [
        {
            "name": "ecs_surface_d14_public_reject_width_shrink80",
            "region": "east_china_sea",
            "group": "surface",
            "horizon": 14,
            "levels": B.SURFACE_LEVELS,
            "lo_scale": 0.90,
            "hi_scale": 0.80,
            "anchor_cv_gain": 0.0,
        }
    ]
    B.STATION_DIR_BLOCKS = []
    B.main()

    manifest = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    manifest["candidate_type"] = "isolated_large_gap_surface_speed_width_shrink"
    manifest["public_feedback_basis"] = {
        "current_confirmed_base": "sub_ecsp14s70.zip",
        "target_metric": "WS ECS Surface d14",
        "target_reason": "Second-largest remaining raw speed gap versus the 2026-06-12 JLShen row.",
        "rejected_prior_branches": [
            {
                "submission": "sub_ecs_sfc14_anlg.zip",
                "public_change": "11.3228 -> 12.2264",
                "interpretation": "Analog center replacement was public-negative.",
            },
            {
                "submission": "sub_ecs_sfc14_hres25_gate.zip",
                "public_change": "11.3228 -> 11.9700",
                "interpretation": "HRES blend/center replacement was public-negative.",
            },
            {
                "submission": "sub_biggap_wcal.zip",
                "public_change": "11.3228 -> 11.6762",
                "interpretation": "Slight width widening was public-negative.",
            },
        ],
        "chosen_shape": {
            "lo_scale": 0.90,
            "hi_scale": 0.80,
            "reason": (
                "Center models and slight widening were public-negative, so this is a q50-locked "
                "shrink probe on the same target."
            ),
        },
        "risk": "No public-positive ECS surface d14 speed block yet; submit as an isolated probe only.",
    }
    manifest["compliance"]["notes"].append(
        "This wrapper applies one q50-locked ECS surface d14 width shrink to the current confirmed base."
    )
    manifest["code_hashes"] = {
        "build_biggap_width_cal_candidate.py": B.sha256(ROOT / "build_biggap_width_cal_candidate.py"),
        "build_ecs_surface_d14_w80_candidate.py": B.sha256(Path(__file__).resolve()),
        "run_ecss14w80_e2e.ps1": B.sha256(ROOT / "run_ecss14w80_e2e.ps1"),
        "run_ecsp14_s70_e2e.ps1": B.sha256(ROOT / "run_ecsp14_s70_e2e.ps1"),
    }
    B.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(B.OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip member validation failed: names={names}, bad={bad}")
    if len(B.OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {B.OUT_ZIP.name}")
    print(f"OK ECS surface d14 shrink-80 candidate: {B.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
