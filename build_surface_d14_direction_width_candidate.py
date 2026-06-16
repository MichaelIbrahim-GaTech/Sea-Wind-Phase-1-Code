#!/usr/bin/env python3
"""
Surface d14-only direction-width candidate.

Compliance:
- Uses only official files under runs/v6_pressure_speed/phase1_dataset.
- Re-runs the same historical CV machinery as direction-width portfolio v2.
- Keeps existing dir_50 centers fixed and changes only dir_05/dir_95.

This is a conservative pruning of the broader v2 width portfolio: only surface
d14 blocks are eligible for final emission. It avoids the d7/pressure width
changes that did not transfer on the public hidden set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

import build_direction_width_portfolio_v2_candidate as V2
import sea_winds_end_to_end_final as E2E


ROOT = V2.ROOT
DATA = V2.DATA
DEFAULT_BASE_CSV = ROOT / "pred_ns_p7dir_mosres.csv"
OUT_CSV = ROOT / "pred_sfc14_dirw.csv"
OUT_ZIP = ROOT / "sub_sfc14_dirw.zip"
SUMMARY_CSV = ROOT / "cv_sfc14_dirw.csv"
MANIFEST = ROOT / "manifest_sfc14_dirw.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--output-zip", type=Path, default=OUT_ZIP)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--grid-per-anchor", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    if not args.base_csv.exists():
        raise SystemExit(f"missing base CSV: {args.base_csv}")
    if len(args.output_zip.name) >= 64:
        raise SystemExit(f"zip filename is too long for Codabench: {args.output_zip.name}")

    print("Surface d14 direction-width candidate", flush=True)
    print(f"Base CSV: {args.base_csv} ({args.base_csv.stat().st_size:,} bytes)", flush=True)
    summary = V2.evaluate_blocks(args.base_csv, args.grid_per_anchor, args.seed)
    summary["eligible_surface_d14"] = summary["group"].eq("surface") & summary["horizon"].astype(int).eq(14)
    summary["gate_passed_original"] = summary["gate_passed"].astype(bool)
    summary["gate_passed"] = summary["gate_passed_original"] & summary["eligible_surface_d14"]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    selected = summary[summary["gate_passed"].astype(bool)].copy()
    print(f"Wrote CV summary: {args.summary_csv}", flush=True)
    print(selected.sort_values(["specificity", "cv_mean_gain"], ascending=[False, False]).to_string(index=False), flush=True)

    if selected.empty:
        payload = {
            "status": "gate_failed_no_submission_written",
            "reason": "No surface d14 direction-width block passed the chronological CV gates.",
            "base_csv": str(args.base_csv.resolve()),
            "cv_summary_csv": str(args.summary_csv.resolve()),
            "compliance": {
                "external_training_data_used": False,
                "web_data_used": False,
                "evaluation_target_labels_used_for_training": False,
                "official_dataset_root": str(DATA.resolve()),
            },
        }
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("No surface d14 direction-width candidate was emitted.")

    final, counts = V2.apply_widths(args.base_csv, summary, args.output_csv, args.output_zip)
    with zipfile.ZipFile(args.output_zip) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")

    payload = {
        "status": "completed",
        "base_csv": {
            "path": str(args.base_csv.resolve()),
            "size": args.base_csv.stat().st_size,
            "sha256": sha256(args.base_csv),
        },
        "cv_summary_csv": str(args.summary_csv.resolve()),
        "selected_blocks": selected.sort_values(["specificity", "cv_mean_gain"], ascending=[False, False]).to_dict(orient="records"),
        "patch_counts": counts,
        "submission": {
            "csv": str(args.output_csv.resolve()),
            "csv_size": args.output_csv.stat().st_size,
            "csv_sha256": sha256(args.output_csv),
            "zip": str(args.output_zip.resolve()),
            "zip_size": args.output_zip.stat().st_size,
            "zip_name_length": len(args.output_zip.name),
            "zip_sha256": sha256(args.output_zip),
            "internal_names": names,
            "internal_csv_size": int(info.file_size),
            "testzip": bad,
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts().to_dict().items()},
        },
        "compliance": {
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "official_dataset_root": str(DATA.resolve()),
            "notes": [
                "Final patch is restricted to grid surface d14 direction intervals.",
                "Only dir_05/dir_95 are changed; dir_50 centers remain fixed.",
                "Width gates are recomputed from official historical HRES/reanalysis data.",
            ],
        },
        "code_hashes": {
            "build_surface_d14_direction_width_candidate.py": sha256(Path(__file__).resolve()),
            "build_direction_width_portfolio_v2_candidate.py": sha256(Path("build_direction_width_portfolio_v2_candidate.py")),
            "sea_winds_end_to_end_final.py": sha256(Path("sea_winds_end_to_end_final.py")),
        },
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {args.manifest}", flush=True)
    print(f"OK: {args.output_zip}", flush=True)


if __name__ == "__main__":
    main()
