from __future__ import annotations

import json
import zipfile
from pathlib import Path

import build_station_dir_width_expand_candidate as W


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def main() -> None:
    W.OUT_CSV = WORK / "pred_ecs14_w150.csv"
    W.OUT_ZIP = WORK / "sub_ecs14_w150.zip"
    W.MANIFEST = WORK / "manifest_ecs14_w150.json"
    W.WIDTH_BLOCKS = [
        {
            "name": "ecs_station_d14_widen_150",
            "region": "east_china_sea",
            "horizon": 14,
            "target_half_width": 150.0,
            "public_tighten_result": "150-degree half-width improved public score to 323.6339 inside sub_stndir_wexp.zip; NS width expansions were negative and are excluded.",
        }
    ]
    W.main()

    manifest = json.loads(W.MANIFEST.read_text(encoding="utf-8"))
    manifest["public_feedback_basis"] = {
        "source_submission": "sub_stndir_wexp.zip",
        "kept_public_positive_blocks": ["ecs_station_d14_widen_150"],
        "dropped_public_negative_blocks": ["ns_station_d7_widen", "ns_station_d14_widen"],
        "reason": "Isolate the only public-positive station-direction width expansion on top of sub_speedpos_wcal.",
    }
    manifest["code_hashes"]["build_ecs_stn14_w150_candidate.py"] = W.sha256(Path(__file__).resolve())
    manifest["code_hashes"]["run_ecs_stn14_w150_e2e.ps1"] = W.sha256(ROOT / "run_ecs_stn14_w150_e2e.ps1")
    W.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(W.OUT_ZIP) as zf:
        names = zf.namelist()
    if names != ["predictions.csv"]:
        raise SystemExit(f"zip member validation failed: {names}")
    print(f"OK ECS station d14 width-150 candidate: {W.OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
