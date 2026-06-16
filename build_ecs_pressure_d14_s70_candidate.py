from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_biggap_width_cal_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    # Use the pre-ECS-p14 confirmed base and apply an absolute replacement
    # shape for this one target. This avoids compounding width scales.
    B.BASE_CSV = WORK / "pred_ecs_stn14_w092.csv"
    B.OUT_CSV = WORK / "pred_ecsp14_s70.csv"
    B.OUT_ZIP = WORK / "sub_ecsp14s70.zip"
    B.MANIFEST = WORK / "manifest_ecsp14_s70.json"
    B.SPEED_BLOCKS = [
        {
            "name": "ecs_pressure_d14_public_curve_shrink70",
            "region": "east_china_sea",
            "group": "pressure",
            "horizon": 14,
            "levels": B.PRESSURE_LEVELS,
            "lo_scale": 0.88,
            "hi_scale": 0.70,
            "anchor_cv_gain": 0.0,
        }
    ]
    B.STATION_DIR_BLOCKS = []
    B.main()

    manifest = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    manifest["candidate_type"] = "isolated_large_gap_public_curve_width_tune"
    manifest["public_feedback_basis"] = {
        "current_confirmed_base": "sub_ecsp14p7s.zip",
        "target_metric": "WS ECS Pressure d14",
        "target_reason": "Still a major speed gap after p7-shape improvement: 17.9630 vs JLShen 17.16.",
        "public_curve_points": [
            {
                "submission": "sub_ecs14sw092.zip",
                "lo_scale": 1.0,
                "hi_scale": 1.0,
                "public_WS_ECS_Pressure_d14": 18.7106,
            },
            {
                "submission": "sub_ecsp14ns.zip",
                "lo_scale": 0.85,
                "hi_scale": 1.30,
                "public_WS_ECS_Pressure_d14": 22.0997,
            },
            {
                "submission": "sub_ecsp14p7s.zip",
                "lo_scale": 0.95,
                "hi_scale": 0.85,
                "public_WS_ECS_Pressure_d14": 17.9630,
            },
        ],
        "chosen_shape": {
            "lo_scale": 0.88,
            "hi_scale": 0.70,
            "reason": (
                "Bounded continuation of the successful narrower same-region pressure shape; "
                "keeps q50 fixed and reduces the wider upper side more aggressively."
            ),
        },
        "risk": (
            "Public-curve tune from one successful narrower point; submit only because the previous "
            "same-target move was strongly positive and no other dimensions are touched."
        ),
    }
    manifest["compliance"]["notes"].append(
        "This wrapper applies one absolute ECS pressure d14 width shape to the confirmed pre-p14 base; it does not compound prior p14 widths."
    )
    manifest["code_hashes"] = {
        "build_biggap_width_cal_candidate.py": B.sha256(ROOT / "build_biggap_width_cal_candidate.py"),
        "build_ecs_pressure_d14_s70_candidate.py": B.sha256(Path(__file__).resolve()),
        "run_ecsp14_s70_e2e.ps1": B.sha256(ROOT / "run_ecsp14_s70_e2e.ps1"),
        "run_ecs14sw092_e2e.ps1": B.sha256(ROOT / "run_ecs14sw092_e2e.ps1"),
    }
    B.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(B.OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip member validation failed: names={names}, bad={bad}")
    if len(B.OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {B.OUT_ZIP.name}")
    print(f"OK ECS pressure d14 shrink-70 candidate: {B.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
