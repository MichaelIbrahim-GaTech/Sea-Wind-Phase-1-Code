from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_seasonal_backtest_gate_candidate as B
from build_seasonal_profile_gate_candidate import parse_weight


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def parse_block(raw: str) -> tuple[str, str, int, str]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "block must be region,group,horizon,candidate "
            "(example: north_sea,pressure,14,blend_seasonal_w21_0.75)"
        )
    region, group, horizon, candidate = parts
    return region, group, int(horizon), candidate


def make_select_blocks(specs: list[tuple[str, str, int, str]]):
    spec_set = set(specs)

    def select_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
        B.require(B.BACKTEST_CSV, "Run seasonal_direction_backtest.py against the selected base CSV first.")
        df = pd.read_csv(B.BACKTEST_CSV)
        current = df[df["candidate"].eq("current_model")][
            ["region", "group", "horizon", "score", "half_width"]
        ].rename(columns={"score": "current_score", "half_width": "current_half_width"})
        cand = df[~df["candidate"].eq("current_model")].merge(
            current, on=["region", "group", "horizon"], how="left", validate="many_to_one"
        )
        cand["gain_vs_current"] = cand["current_score"] - cand["score"]
        cand["weight_current"] = cand["candidate"].map(parse_weight)
        cand["public_gap"] = [
            float(B.PUBLIC_CURRENT[(str(r.region), str(r.group), int(r.horizon))]
                  - B.PUBLIC_TOP5_BEST[(str(r.region), str(r.group), int(r.horizon))])
            for r in cand.itertuples(index=False)
        ]
        cand["gate_passed"] = [
            (str(r.region), str(r.group), int(r.horizon), str(r.candidate)) in spec_set
            for r in cand.itertuples(index=False)
        ]
        selected_parts = []
        for spec in specs:
            region, group, horizon, candidate = spec
            row = cand[
                cand["region"].eq(region)
                & cand["group"].eq(group)
                & cand["horizon"].eq(horizon)
                & cand["candidate"].eq(candidate)
            ]
            if len(row) != 1:
                raise SystemExit(f"Could not find unique backtest row for fixed seasonal block: {spec}")
            selected_parts.append(row)
        selected = pd.concat(selected_parts, ignore_index=True)
        cand = cand.sort_values(
            ["gate_passed", "region", "group", "horizon", "gain_vs_current", "score"],
            ascending=[False, True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return cand, selected.reset_index(drop=True)

    return select_blocks


def update_manifest(args: argparse.Namespace, specs: list[tuple[str, str, int, str]]) -> None:
    if not B.MANIFEST.exists():
        return
    payload = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    payload["fixed_block_selection"] = {
        "profile": args.profile,
        "blocks": [
            {"region": r, "group": g, "horizon": h, "candidate": c}
            for r, g, h, c in specs
        ],
        "reason": args.reason,
        "public_feedback_used_for_model_selection": True,
        "public_feedback_note": (
            "The fixed block list is based on public leaderboard feedback from the generated "
            "sub_seas_b75.zip branch. Final predictions are still generated from official "
            "competition data and generated official-data artifacts only."
        ),
    }
    payload["code_hashes"]["build_seasonal_fixed_blocks_candidate.py"] = B.sha256(Path(__file__).resolve())
    B.MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="nspos_b75")
    ap.add_argument("--reason", default="Fixed public-positive seasonal blocks from blend75 feedback.")
    ap.add_argument("--base-csv", type=Path, default=WORK / "pred_sfc14_selw.csv")
    ap.add_argument("--backtest-csv", type=Path, default=WORK / "seasonal_direction_backtest_on_sfc14_selw.csv")
    ap.add_argument("--out-csv", type=Path, default=WORK / "pred_nspos_b75.csv")
    ap.add_argument("--out-zip", type=Path, default=WORK / "sub_nspos_b75.zip")
    ap.add_argument("--manifest", type=Path, default=WORK / "manifest_nspos_b75.json")
    ap.add_argument("--block", action="append", type=parse_block, required=True)
    args = ap.parse_args()

    B.BASE_CSV = args.base_csv
    B.BACKTEST_CSV = args.backtest_csv
    B.OUT_CSV = args.out_csv
    B.OUT_ZIP = args.out_zip
    B.MANIFEST = args.manifest
    B.select_blocks = make_select_blocks(args.block)

    B.main()
    update_manifest(args, args.block)


if __name__ == "__main__":
    main()
