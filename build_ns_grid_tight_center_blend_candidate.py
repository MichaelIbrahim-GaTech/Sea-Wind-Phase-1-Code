from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_ns_grid_dir_regime_fast_gate_candidate as FG
import hres_mos_residual_branch as HM
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_ns_p7dir_mosres.csv"
OUT_CSV = WORK / "pred_ns_grid_tight_center_blend.csv"
OUT_ZIP = WORK / "sub_ns_ctblend.zip"
MANIFEST = WORK / "manifest_ns_grid_tight_center_blend.json"
CV_BY_FOLD = WORK / "cv_ns_grid_tight_center_blend_by_fold.csv"
CV_SUMMARY = WORK / "cv_ns_grid_tight_center_blend_summary.csv"

# Small analog/HRES center moves only. The previous regime analog branch used
# 0.75 and had hidden-public instability; this branch tests modest blends and
# then enforces an inference-movement gate before writing any submission.
TIGHT_BLOCKS = (("surface", 14), ("pressure", 14))
TIGHT_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

PUBLIC_MEAN_MARGIN = {
    ("surface", 14): 4.0,
    ("pressure", 14): 3.0,
}
PUBLIC_MAX_SLACK = {
    ("surface", 14): 5.0,
    ("pressure", 14): 2.5,
}
MIN_BASELINE_MEAN_GAIN = {
    ("surface", 14): -4.0,
    ("pressure", 14): 4.0,
}
MAX_CENTER_SHIFT_MEAN = 24.0
MAX_CENTER_SHIFT_P90 = 60.0
MAX_CENTER_SHIFT_P99 = 105.0

COLS = E2E.COLS
DIR_COLS = E2E.DIR_COLS
SPEED_COLS = E2E.SPEED_COLS


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_member(zip_path: Path, member: str = "predictions.csv") -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def cleanup_outputs() -> None:
    for path in (OUT_CSV, OUT_ZIP, MANIFEST, CV_BY_FOLD, CV_SUMMARY):
        if path.exists():
            path.unlink()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def configure_imported_factory() -> None:
    FG.BLEND_WEIGHTS_ANALOG = TIGHT_WEIGHTS
    FG.BLOCKS = TIGHT_BLOCKS
    FG.CV_BY_FOLD = CV_BY_FOLD
    FG.CV_SUMMARY = CV_SUMMARY


def parse_weight(candidate: str) -> float:
    mode, _, _, weight = FG.parse_candidate(candidate)
    return float(weight) if mode == "blend" else 1.0


def selection_reason(row: pd.Series) -> tuple[bool, str]:
    key = (str(row["group"]), int(row["horizon"]))
    candidate = str(row["candidate"])
    if candidate == "hres":
        return False, "hres_baseline_not_a_center_blend"
    if parse_weight(candidate) > 0.50:
        return False, "blend_weight_above_0.50_rejected_after_public_instability"

    score_mean = float(row["score_mean"])
    score_max = float(row["score_max"])
    public_current = float(row["public_current"])
    baseline_gain = float(row["mean_gain_vs_baseline_ref"])
    public_margin = public_current - score_mean
    max_slack = score_max - public_current

    reasons = []
    if public_margin < PUBLIC_MEAN_MARGIN[key]:
        reasons.append(f"public_margin_{public_margin:.3f}_below_{PUBLIC_MEAN_MARGIN[key]:.3f}")
    if max_slack > PUBLIC_MAX_SLACK[key]:
        reasons.append(f"score_max_slack_{max_slack:.3f}_above_{PUBLIC_MAX_SLACK[key]:.3f}")
    if baseline_gain < MIN_BASELINE_MEAN_GAIN[key]:
        reasons.append(f"baseline_mean_gain_{baseline_gain:.3f}_below_{MIN_BASELINE_MEAN_GAIN[key]:.3f}")
    if reasons:
        return False, "; ".join(reasons)
    return True, "accepted_by_public_margin_cv_and_weight_gate"


def selected_blocks(summary: pd.DataFrame) -> tuple[list[FG.SelectedBlock], pd.DataFrame]:
    decisions = []
    selected: list[FG.SelectedBlock] = []
    for _, row in summary.iterrows():
        ok, reason = selection_reason(row)
        d = row.to_dict()
        d["tight_gate_passed"] = bool(ok)
        d["tight_gate_reason"] = reason
        decisions.append(d)

    decisions_df = pd.DataFrame(decisions).sort_values(
        ["tight_gate_passed", "group", "horizon", "score_mean", "score_max"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )

    for group, horizon in TIGHT_BLOCKS:
        sub = decisions_df[
            decisions_df["group"].eq(group)
            & decisions_df["horizon"].astype(int).eq(int(horizon))
            & decisions_df["tight_gate_passed"].astype(bool)
        ].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["score_mean", "score_max"], kind="mergesort").iloc[0]
        selected.append(
            FG.SelectedBlock(
                group=group,
                horizon=int(horizon),
                candidate=str(best["candidate"]),
                half_width=float(best["half_width_mean"]),
                score_mean=float(best["score_mean"]),
                score_max=float(best["score_max"]),
                baseline_ref=float(best["baseline_ref"]),
                public_current=float(best["public_current"]),
            )
        )
    return selected, decisions_df


def movement_gate(stability: list[dict[str, object]]) -> tuple[bool, list[str]]:
    failed = []
    for item in stability:
        mean = float(item["center_delta_mean"])
        p90 = float(item["center_delta_p90"])
        p99 = float(item["center_delta_p99"])
        if mean > MAX_CENTER_SHIFT_MEAN or p90 > MAX_CENTER_SHIFT_P90 or p99 > MAX_CENTER_SHIFT_P99:
            failed.append(
                f"{item['block']}: mean={mean:.2f} p90={p90:.2f} p99={p99:.2f} "
                f"limits=({MAX_CENTER_SHIFT_MEAN:.1f},{MAX_CENTER_SHIFT_P90:.1f},{MAX_CENTER_SHIFT_P99:.1f})"
            )
    return not failed, failed


def zip_payload() -> dict[str, object]:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")
    return {
        "csv": str(OUT_CSV),
        "zip": str(OUT_ZIP),
        "zip_name_length": len(OUT_ZIP.name),
        "csv_size": int(OUT_CSV.stat().st_size),
        "zip_size": int(OUT_ZIP.stat().st_size),
        "csv_sha256": sha256(OUT_CSV),
        "zip_sha256": sha256(OUT_ZIP),
        "internal_names": names,
        "internal_csv_size": int(info.file_size),
        "internal_csv_sha256": sha256_zip_member(OUT_ZIP),
        "testzip": bad,
    }


def write_manifest(
    status: str,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    blocks: list[FG.SelectedBlock],
    reason: str,
    stability: list[dict[str, object]] | None = None,
    delta: dict[str, object] | None = None,
    patch_counts: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": status,
        "reason": reason,
        "submission": zip_payload() if OUT_ZIP.exists() else None,
        "base_csv": {
            "path": str(BASE_CSV),
            "size": int(BASE_CSV.stat().st_size) if BASE_CSV.exists() else None,
            "sha256": sha256(BASE_CSV) if BASE_CSV.exists() else None,
        },
        "accepted_blocks": [block.__dict__ for block in blocks],
        "patch_counts": patch_counts or {},
        "inference_stability": stability or [],
        "delta": delta or {},
        "cv_by_fold_csv": str(CV_BY_FOLD),
        "cv_summary_csv": str(CV_SUMMARY),
        "cv_summary": summary.to_dict(orient="records"),
        "tight_decisions": decisions.to_dict(orient="records"),
        "gates": {
            "blocks": [f"{g}_d{h}" for g, h in TIGHT_BLOCKS],
            "blend_weights": list(TIGHT_WEIGHTS),
            "public_mean_margin": {f"{g}_d{h}": v for (g, h), v in PUBLIC_MEAN_MARGIN.items()},
            "public_max_slack": {f"{g}_d{h}": v for (g, h), v in PUBLIC_MAX_SLACK.items()},
            "min_baseline_mean_gain": {f"{g}_d{h}": v for (g, h), v in MIN_BASELINE_MEAN_GAIN.items()},
            "max_center_shift_mean": MAX_CENTER_SHIFT_MEAN,
            "max_center_shift_p90": MAX_CENTER_SHIFT_P90,
            "max_center_shift_p99": MAX_CENTER_SHIFT_P99,
        },
        "compliance": FG.compliance_payload(),
        "code_hashes": {
            "build_ns_grid_tight_center_blend_candidate.py": sha256(Path(__file__).resolve()),
            "build_ns_grid_dir_regime_fast_gate_candidate.py": sha256(ROOT / "build_ns_grid_dir_regime_fast_gate_candidate.py"),
            "hres_mos_residual_branch.py": sha256(ROOT / "hres_mos_residual_branch.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(payload, indent=2, sort_keys=True))
    log(f"Wrote {MANIFEST}")


def run_cv() -> pd.DataFrame:
    feat = FG.load_train_features()
    grid = FG.build_grid(feat)
    feat = FG.attach_grid_idx(feat, grid)
    means = FG.feature_daily_means(feat)
    store = FG.build_surface_store(feat, grid)
    cube, _ = FG.load_pressure_cube()
    cube_idx_map = FG.pressure_index_map(grid, cube)

    _, summary = FG.run_cv(feat, means, store, cube, cube_idx_map)
    return feat, grid, means, store, cube, cube_idx_map, summary


def main() -> None:
    require(BASE_CSV, "Run .\\run_ns_p7dir_mosres_e2e.ps1 first.")
    cleanup_outputs()
    configure_imported_factory()

    feat, grid, means, store, cube, cube_idx_map, summary = run_cv()
    blocks, decisions = selected_blocks(summary)
    decisions.to_csv(WORK / "decision_ns_grid_tight_center_blend.csv", index=False)

    if not blocks:
        write_manifest(
            "gate_failed_no_submission_written",
            summary,
            decisions,
            [],
            "No small center-blend candidate cleared the tight CV/public-margin gates.",
        )
        return

    log(f"Selected tight center blocks: {[block.__dict__ for block in blocks]}")
    base = FG.normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    before = base.copy()
    patched, patch_counts, stability = FG.apply_blocks(base, blocks, grid, means, store, cube, cube_idx_map)
    stable, failures = movement_gate(stability)
    if not stable:
        write_manifest(
            "gate_failed_no_submission_written",
            summary,
            decisions,
            blocks,
            "Selected blocks failed final inference center-shift gate: " + " | ".join(failures),
            stability=stability,
            patch_counts=patch_counts,
        )
        return

    final = E2E.validate_final(patched)
    delta = FG.validate_delta(before, final, blocks)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(
        "submission_written_after_tight_cv_and_inference_movement_gates",
        summary,
        decisions,
        blocks,
        "All gates passed.",
        stability=stability,
        delta=delta,
        patch_counts=patch_counts,
    )
    log(f"OK: {OUT_ZIP}")


if __name__ == "__main__":
    main()
