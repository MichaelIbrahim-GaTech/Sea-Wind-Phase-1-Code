from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_ecs_stn14_speed_w108_candidate as W


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    W.WIDTH_SCALE = 0.92
    W.OUT_CSV = WORK / "pred_ecs_stn14_w092.csv"
    W.OUT_ZIP = WORK / "sub_ecs14sw092.zip"
    W.MANIFEST = WORK / "manifest_ecs_stn14_w092.json"
    W.main()

    manifest = json.loads(W.MANIFEST.read_text(encoding="utf-8"))
    manifest["public_feedback_basis"] = {
        "base_submission": "sub_ns14sw1145.zip",
        "rejected_prior": "sub_ecs14sw108.zip",
        "rejected_prior_result": {
            "WS_ECS_Stations_d14": "9.0772 -> 9.2888",
            "interpretation": "Width-only expansion was public-negative, so this branch tests narrower ECS station d14 speed intervals while keeping q50 unchanged.",
        },
        "chosen_scale": 0.92,
    }
    manifest["code_hashes"] = {
        "build_ecs_stn14_speed_w108_candidate.py": W.sha256(ROOT / "build_ecs_stn14_speed_w108_candidate.py"),
        "build_ecs_stn14_speed_w092_candidate.py": W.sha256(Path(__file__).resolve()),
        "run_ecs14sw092_e2e.ps1": W.sha256(ROOT / "run_ecs14sw092_e2e.ps1"),
    }
    W.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(W.OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"zip member validation failed: names={names}, bad={bad}")
    print(f"OK ECS station d14 speed width-0.92 candidate: {W.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
