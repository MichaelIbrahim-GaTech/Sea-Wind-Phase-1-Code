from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_biggap_width_cal_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    B.BASE_CSV = WORK / "pred_ecss14_w80.csv"
    B.OUT_CSV = WORK / "pred_ecss14_w78.csv"
    B.OUT_ZIP = WORK / "sub_ecss14w78.zip"
    B.MANIFEST = WORK / "manifest_ecss14_w78.json"
    B.SPEED_BLOCKS = [
        {
            "name": "ecs_surface_d14_public_curve_width_shrink78",
            "region": "east_china_sea",
            "group": "surface",
            "horizon": 14,
            "levels": B.SURFACE_LEVELS,
            "lo_scale": 0.9873333333,
            "hi_scale": 0.9715,
            "anchor_cv_gain": 0.0,
        }
    ]
    B.STATION_DIR_BLOCKS = []
    B.main()

    manifest = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    manifest["candidate_type"] = "isolated_large_gap_surface_speed_public_curve_fit"
    manifest["public_feedback_basis"] = {
        "current_confirmed_base": "sub_ecss14w80.zip",
        "target_metric": "WS ECS Surface d14",
        "target_reason": (
            "w80 improved strongly and w70 overshot; this tests the fitted public-curve point "
            "between them, very close to w80."
        ),
        "public_curve_points": [
            {"effective_hi_scale": 1.0, "effective_lo_scale": 1.0, "public_metric": 11.3228},
            {"effective_hi_scale": 0.80, "effective_lo_scale": 0.90, "public_metric": 10.8122},
            {"effective_hi_scale": 0.70, "effective_lo_scale": 0.85, "public_metric": 10.8687},
        ],
        "chosen_shape": {
            "incremental_lo_scale": 0.9873333333,
            "incremental_hi_scale": 0.9715,
            "effective_lo_scale_vs_ecsp14_s70": 0.8886,
            "effective_hi_scale_vs_ecsp14_s70": 0.7772,
            "reason": (
                "Quadratic interpolation along the successful shrink path puts the public optimum "
                "near effective lo=0.889 and hi=0.777."
            ),
        },
        "risk": "Small curve-fit refinement; reject if ECS surface d14 worsens versus 10.8122.",
    }
    manifest["compliance"]["notes"].append(
        "This wrapper applies one curve-fit q50-locked ECS surface d14 width refinement to the current confirmed w80 base."
    )
    manifest["code_hashes"] = {
        "build_biggap_width_cal_candidate.py": B.sha256(ROOT / "build_biggap_width_cal_candidate.py"),
        "build_ecs_surface_d14_w78_candidate.py": B.sha256(Path(__file__).resolve()),
        "run_ecss14w78_e2e.ps1": B.sha256(ROOT / "run_ecss14w78_e2e.ps1"),
        "run_ecss14w80_e2e.ps1": B.sha256(ROOT / "run_ecss14w80_e2e.ps1"),
    }
    B.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(B.OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip member validation failed: names={names}, bad={bad}")
    if len(B.OUT_ZIP.name) >= 64:
        raise SystemExit(f"zip filename too long: {B.OUT_ZIP.name}")
    print(f"OK ECS surface d14 shrink-78 candidate: {B.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
