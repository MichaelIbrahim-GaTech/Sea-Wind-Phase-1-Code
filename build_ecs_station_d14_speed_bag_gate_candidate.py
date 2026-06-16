from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_station_speed_bag_gate_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip structure/testzip: names={names}, bad={bad}")


def patch_manifest() -> None:
    manifest = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    manifest["public_feedback_basis"] = {
        "promoted_base_zip": "sub_ns14sw1145.zip",
        "large_gap_target": "WS ECS Stations d14",
        "base_public_score": 9.0772,
        "leader_public_score": 8.84,
        "reason": "Test an untried station d14 speed target instead of continuing saturated direction-width changes.",
    }
    manifest["code_hashes"]["build_ecs_station_d14_speed_bag_gate_candidate.py"] = B.sha256(Path(__file__).resolve())
    B.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    B.BASE_CSV = WORK / "pred_ns_stn14_w1145.csv"
    B.OUT_CSV = WORK / "pred_ecs_stn14spd_bag.csv"
    B.OUT_ZIP = WORK / "sub_ecs14spd_bag.zip"
    B.SUMMARY_CSV = WORK / "cv_ecs14spd_bag.csv"
    B.MANIFEST = WORK / "manifest_ecs14spd_bag.json"
    B.TARGETS = [
        {
            "id": "ecs_d14",
            "label": "WS ECS Stations d14",
            "region": "east_china_sea",
            "horizon": 14,
            "expected_rows": 224,
            "absolute_score_mean_lte": 9.25,
            "absolute_score_max_lte": 10.25,
            "public_current": 9.0772,
            "families": ["direct_bag2", "hres_resid_bag2", "recent14_resid_bag2", "log_bag2"],
        }
    ]
    B.main()
    patch_manifest()
    validate_zip(B.OUT_ZIP)
    print(f"OK: {B.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
