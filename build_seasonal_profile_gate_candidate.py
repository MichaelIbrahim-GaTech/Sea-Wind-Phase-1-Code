from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_seasonal_backtest_gate_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"


def parse_weight(candidate: str) -> float | None:
    _, weight_current = B.parse_candidate(candidate)
    return weight_current


def make_select_blocks(
    min_backtest_gain: float,
    min_public_gap: float,
    min_current_weight: float | None,
):
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
        public_gaps = []
        gate = []
        for row in cand.itertuples(index=False):
            key = (str(row.region), str(row.group), int(row.horizon))
            gap = float(B.PUBLIC_CURRENT[key] - B.PUBLIC_TOP5_BEST[key])
            public_gaps.append(gap)
            weight_ok = True
            if min_current_weight is not None:
                weight_ok = row.weight_current is not None and float(row.weight_current) >= float(min_current_weight)
            gate.append(
                bool(
                    gap >= min_public_gap
                    and float(row.gain_vs_current) >= min_backtest_gain
                    and weight_ok
                )
            )
        cand["public_gap"] = public_gaps
        cand["gate_passed"] = gate
        cand = cand.sort_values(
            ["gate_passed", "region", "group", "horizon", "gain_vs_current", "score"],
            ascending=[False, True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        selected = cand[cand["gate_passed"].astype(bool)].groupby(
            ["region", "group", "horizon"], as_index=False
        ).head(1)
        selected = selected.reset_index(drop=True)
        return cand, selected

    return select_blocks


def update_manifest(args: argparse.Namespace) -> None:
    if not B.MANIFEST.exists():
        return
    payload = json.loads(B.MANIFEST.read_text(encoding="utf-8"))
    payload["profile_gate"] = {
        "profile": args.profile,
        "min_backtest_gain": args.min_backtest_gain,
        "min_public_gap": args.min_public_gap,
        "min_current_weight": args.min_current_weight,
        "description": (
            "Seasonal candidates are selected from the official-data historical backtest, "
            "then restricted by this profile before full end-to-end generation."
        ),
    }
    payload["code_hashes"]["build_seasonal_profile_gate_candidate.py"] = B.sha256(Path(__file__).resolve())
    B.MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="b75")
    ap.add_argument("--base-csv", type=Path, default=WORK / "pred_sfc14_selw.csv")
    ap.add_argument("--backtest-csv", type=Path, default=WORK / "seasonal_direction_backtest_on_sfc14_selw.csv")
    ap.add_argument("--out-csv", type=Path, default=WORK / "pred_seas_b75.csv")
    ap.add_argument("--out-zip", type=Path, default=WORK / "sub_seas_b75.zip")
    ap.add_argument("--manifest", type=Path, default=WORK / "manifest_seas_b75.json")
    ap.add_argument("--min-backtest-gain", type=float, default=7.5)
    ap.add_argument("--min-public-gap", type=float, default=8.0)
    ap.add_argument(
        "--min-current-weight",
        type=float,
        default=0.75,
        help="Require blend candidates to retain at least this weight on the current center.",
    )
    args = ap.parse_args()

    B.BASE_CSV = args.base_csv
    B.BACKTEST_CSV = args.backtest_csv
    B.OUT_CSV = args.out_csv
    B.OUT_ZIP = args.out_zip
    B.MANIFEST = args.manifest
    B.MIN_BACKTEST_GAIN = args.min_backtest_gain
    B.MIN_PUBLIC_GAP = args.min_public_gap
    B.select_blocks = make_select_blocks(
        min_backtest_gain=args.min_backtest_gain,
        min_public_gap=args.min_public_gap,
        min_current_weight=args.min_current_weight,
    )

    B.main()
    update_manifest(args)


if __name__ == "__main__":
    main()
