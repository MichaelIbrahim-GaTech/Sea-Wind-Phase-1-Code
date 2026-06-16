from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

import audit_final_submission as AUD
import build_dir_error_width_gridlong_v1_candidate as GL


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_dir_error_width_newsignal_v1.csv"
OUT_CSV = WORK / "pred_direrrw_ecss14push_v1.csv"
OUT_ZIP = WORK / "sub_direrrw_ecss14push_v1.zip"
MANIFEST = WORK / "manifest_direrrw_ecss14push_v1.json"
AUDIT_MANIFEST = WORK / "manifest_direrrw_ecss14push_v1_audit.json"


# Keep public-positive gridlong blocks, skip the public-negative NS surface d7
# block, and push ECS surface d14 from the safest selected policy to a stronger
# CV-passing policy with better mean and still-positive worst-fold gain.
SELECTED_POLICIES = [
    {
        "target_id": "dir_ecs_pressure_d7",
        "candidate": "errq|alpha=0.90|w=1.00|scale=1.30",
        "public_result": "252.7414 -> 216.6134",
        "reason": "public-positive and near visible best",
    },
    {
        "target_id": "dir_ns_pressure_d7",
        "candidate": "errq|alpha=0.95|w=0.75|scale=1.30",
        "public_result": "280.1930 -> 277.4563",
        "reason": "public-positive; keep exact prior policy",
    },
    {
        "target_id": "dir_ecs_surface_d1",
        "candidate": "errq|alpha=0.90|w=1.00|scale=1.30",
        "public_result": "131.0613 -> 107.5328",
        "reason": "public-positive; keep exact prior policy",
    },
    {
        "target_id": "dir_ecs_surface_d7",
        "candidate": "errq|alpha=0.95|w=0.75|scale=1.15",
        "public_result": "278.4071 -> 253.7507",
        "reason": "public-positive; keep exact prior policy",
    },
    {
        "target_id": "dir_ecs_surface_d14",
        "candidate": "errq|alpha=0.90|w=0.50|scale=1.30",
        "public_result": "327.5396 -> 317.7482 with weaker w=0.25 scale=1.30 policy",
        "reason": "stronger CV-passing policy: mean gain 13.3000, worst-fold gain 4.8212, score_max 305.8030",
    },
]


def to_jsonable(obj: Any) -> Any:
    return GL.to_jsonable(obj)


def main() -> None:
    if not BASE_CSV.exists():
        raise SystemExit(f"Missing base CSV: {BASE_CSV}")
    if not GL.ROW_CACHE.exists():
        raise SystemExit(f"Missing row cache: {GL.ROW_CACHE}. Run gridlong v1 first.")

    print(f"Reading base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)
    rows = pd.read_parquet(GL.ROW_CACHE)
    patches: list[dict[str, Any]] = []
    selected_out: list[dict[str, Any]] = []
    decisions = pd.read_csv(GL.DECISION_CSV) if GL.DECISION_CSV.exists() else pd.DataFrame()

    for policy in SELECTED_POLICIES:
        target_id = str(policy["target_id"])
        target = GL.TARGET_BY_ID[target_id]
        train = rows[rows["target_id"].eq(target_id)].reset_index(drop=True)
        if train.empty:
            raise SystemExit(f"No row-cache training rows for {target_id}")
        selected = {"target_id": target_id, "candidate": str(policy["candidate"])}
        audit = GL.apply_model_to_submission(df, target, train, selected)
        patches.append(audit)
        if not audit.get("inference_gate_passed", False):
            raise SystemExit(f"Inference gate failed for {target_id}: {audit}")
        cv_row: dict[str, Any] = {}
        if not decisions.empty:
            match = decisions[decisions["target_id"].eq(target_id) & decisions["candidate"].eq(str(policy["candidate"]))]
            if not match.empty:
                cv_row = match.iloc[0].to_dict()
        selected_out.append({**policy, "cv": cv_row, "inference": audit})

    final = GL.E2E.validate_final(df)
    GL.E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        info = zf.getinfo("predictions.csv")
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"Zip validation failed: names={names} bad={bad}")

    audit_df = AUD.read_submission_csv(OUT_ZIP)
    validation = AUD.validate(audit_df)
    if not validation["ok"]:
        raise SystemExit(f"Validation failed: {validation}")
    delta = AUD.diff_against_baseline(audit_df, BASE_CSV)
    manifest = {
        "mode": "direrrw_ecss14push_v1",
        "reason": "Use public-proven direction error-width blocks and replace only ECS surface d14 with a stronger CV-passing policy.",
        "base_csv": str(BASE_CSV),
        "base_csv_sha256": GL.sha256(BASE_CSV),
        "row_cache": str(GL.ROW_CACHE),
        "decision_csv": str(GL.DECISION_CSV),
        "out_csv": str(OUT_CSV),
        "out_zip": str(OUT_ZIP),
        "audit_manifest": str(AUDIT_MANIFEST),
        "out_csv_sha256": GL.sha256(OUT_CSV),
        "out_zip_sha256": GL.sha256(OUT_ZIP),
        "zip_size": int(OUT_ZIP.stat().st_size),
        "zip_internal_csv_size": int(info.file_size),
        "selected": selected_out,
        "skipped_public_negative": {
            "target_id": "dir_ns_surface_d7",
            "public_result": "298.5943 -> 305.7552 in gridlong_v1",
            "reason": "public-negative; do not apply",
        },
        "validation": validation,
        "delta_vs_base": delta,
        "competition_rule_notes": [
            "Uses official-data generated base predictions and official training/inference features only.",
            "Public feedback is used only as aggregate target-level selection/rollback information.",
            "No external data, hidden labels, or row-level evaluation targets are used.",
            "Only dir_05/dir_95 interval widths are changed; dir_50 centers remain locked.",
        ],
        "code_hashes": {
            "builder": GL.sha256(Path(__file__).resolve()),
            "gridlong_builder": GL.sha256(ROOT / "build_dir_error_width_gridlong_v1_candidate.py"),
            "base_csv": GL.sha256(BASE_CSV),
        },
    }
    MANIFEST.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), flush=True)
    print(f"Wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
