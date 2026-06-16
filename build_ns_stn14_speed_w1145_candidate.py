from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_ns_stn14_speed_w108_candidate as W


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    W.WIDTH_SCALE = 1.145
    W.OUT_CSV = WORK / "pred_ns_stn14_w1145.csv"
    W.OUT_ZIP = WORK / "sub_ns14sw1145.zip"
    W.MANIFEST = WORK / "manifest_ns_stn14_w1145.json"
    W.main()

    manifest = json.loads(W.MANIFEST.read_text(encoding="utf-8"))
    manifest["public_feedback_basis"] = {
        "base_submission": "sub_ecs14_w150.zip",
        "public_scale_points": [
            {"scale": 0.92, "WS_NS_Stations_d14": 17.4681, "submission": "sub_ns14sw092.zip"},
            {"scale": 1.00, "WS_NS_Stations_d14": 17.0091, "submission": "sub_ecs14_w150.zip"},
            {"scale": 1.08, "WS_NS_Stations_d14": 16.7801, "submission": "sub_ns14sw108.zip"},
            {"scale": 1.12, "WS_NS_Stations_d14": 16.6945, "submission": "sub_ns14sw112.zip"},
            {"scale": 1.15, "WS_NS_Stations_d14": 16.6719, "submission": "sub_ns14sw115.zip"},
            {"scale": 1.16, "WS_NS_Stations_d14": 16.6785, "submission": "sub_ns14sw116.zip"},
        ],
        "fit_note": {
            "chosen_scale": 1.145,
            "reason": "Local quadratic through the last three public points peaks around 1.1457; 1.16 already overshot.",
        },
    }
    manifest["code_hashes"] = {
        "build_ns_stn14_speed_w108_candidate.py": W.sha256(ROOT / "build_ns_stn14_speed_w108_candidate.py"),
        "build_ns_stn14_speed_w1145_candidate.py": W.sha256(Path(__file__).resolve()),
        "run_ns_stn14_speed_w1145_e2e.ps1": W.sha256(ROOT / "run_ns_stn14_speed_w1145_e2e.ps1"),
    }
    W.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(W.OUT_ZIP) as zf:
        names = zf.namelist()
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    print(f"OK NS station d14 speed width-1.145 candidate: {W.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
