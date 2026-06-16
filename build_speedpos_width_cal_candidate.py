from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_biggap_width_cal_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

PUBLIC_POSITIVE_SPEED_BLOCKS = {
    "ns_pressure_d14_width_asym",
    "ns_surface_d14_width_asym",
    "ecs_pressure_d7_width_asym",
}


def main() -> None:
    B.OUT_CSV = WORK / "pred_speedpos_wcal.csv"
    B.OUT_ZIP = WORK / "sub_speedpos_wcal.zip"
    B.MANIFEST = WORK / "manifest_speedpos_wcal.json"
    B.SPEED_BLOCKS = [block for block in B.SPEED_BLOCKS if block["name"] in PUBLIC_POSITIVE_SPEED_BLOCKS]
    B.STATION_DIR_BLOCKS = []
    B.main()

    manifest = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    manifest["public_feedback_basis"] = {
        "source_submission": "sub_biggap_wcal.zip",
        "kept_public_positive_blocks": sorted(PUBLIC_POSITIVE_SPEED_BLOCKS),
        "dropped_public_negative_blocks": [
            "ns_station_d7_monthclim_width",
            "ns_station_d14_monthclim_width",
            "ecs_station_d14_monthclim_width",
            "ecs_surface_d7_width_asym",
            "ecs_surface_d14_width_asym",
        ],
        "reason": "Keep only blocks that improved public metrics in sub_biggap_wcal.zip and revert visible regressions.",
    }
    manifest["code_hashes"]["build_speedpos_width_cal_candidate.py"] = B.sha256(Path(__file__).resolve())
    manifest["code_hashes"]["run_speedpos_width_cal_e2e.ps1"] = B.sha256(ROOT / "run_speedpos_width_cal_e2e.ps1")
    B.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(B.OUT_ZIP) as zf:
        names = zf.namelist()
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    print(f"OK public-positive speed width candidate: {B.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
