from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_ns_grid_dir_regime_fast_gate_candidate as FG
import build_ns_grid_tight_center_blend_candidate as TCB
import sea_winds_end_to_end_final as E2E


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

BASE_CSV = WORK / "pred_ns_p7dir_mosres.csv"
OUT_CSV = WORK / "pred_ns_sfc14_selective_center.csv"
OUT_ZIP = WORK / "sub_ns_sfc14_sel.zip"
MANIFEST = WORK / "manifest_ns_sfc14_selective_center.json"
CV_BY_FOLD = WORK / "cv_ns_sfc14_selective_center_by_fold.csv"
CV_SUMMARY = WORK / "cv_ns_sfc14_selective_center_summary.csv"
DECISIONS_CSV = WORK / "decision_ns_sfc14_selective_center.csv"

TARGET_GROUP = "surface"
TARGET_HORIZON = 14
ROW_CENTER_SHIFT_MAX = 60.0
MIN_SELECTED_FRACTION = 0.45

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
    for path in (OUT_CSV, OUT_ZIP, MANIFEST, CV_BY_FOLD, CV_SUMMARY, DECISIONS_CSV):
        if path.exists():
            path.unlink()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def configure_factories() -> None:
    FG.BLEND_WEIGHTS_ANALOG = TCB.TIGHT_WEIGHTS
    FG.BLOCKS = ((TARGET_GROUP, TARGET_HORIZON),)
    FG.CV_BY_FOLD = CV_BY_FOLD
    FG.CV_SUMMARY = CV_SUMMARY


def run_cv():
    feat = FG.load_train_features()
    grid = FG.build_grid(feat)
    feat = FG.attach_grid_idx(feat, grid)
    means = FG.feature_daily_means(feat)
    store = FG.build_surface_store(feat, grid)
    cube, _ = FG.load_pressure_cube()
    cube_idx_map = FG.pressure_index_map(grid, cube)
    _, summary = FG.run_cv(feat, means, store, cube, cube_idx_map)
    return grid, means, store, cube, cube_idx_map, summary


def select_surface_block(summary: pd.DataFrame) -> tuple[FG.SelectedBlock | None, pd.DataFrame]:
    blocks, decisions = TCB.selected_blocks(summary)
    decisions.to_csv(DECISIONS_CSV, index=False)
    blocks = [b for b in blocks if b.group == TARGET_GROUP and b.horizon == TARGET_HORIZON]
    if not blocks:
        return None, decisions
    return blocks[0], decisions


def ordered_lookup(base: pd.DataFrame, target_mask: pd.Series, patch: pd.DataFrame) -> pd.DataFrame:
    lookup = base.loc[target_mask].reset_index()[
        ["index", "window", "region", "latitude", "longitude", "horizon", "hour", "level", "dir_05", "dir_50", "dir_95"]
    ]
    merged = lookup.merge(
        patch,
        on=["window", "region", "latitude", "longitude", "horizon", "hour", "level"],
        how="left",
        validate="one_to_one",
    )
    if merged["center"].isna().any():
        raise SystemExit(f"missing candidate centers: {int(merged['center'].isna().sum())}")
    return merged


def base_half_widths(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    center = pd.to_numeric(merged["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
    lo = pd.to_numeric(merged["dir_05"], errors="coerce").to_numpy(dtype="float64") % 360.0
    hi = pd.to_numeric(merged["dir_95"], errors="coerce").to_numpy(dtype="float64") % 360.0
    return (center - lo) % 360.0, (hi - center) % 360.0


def apply_selective_center(
    base: pd.DataFrame,
    block: FG.SelectedBlock,
    grid: pd.DataFrame,
    means: pd.DataFrame,
    store: FG.SurfaceTargetStore,
    cube,
    cube_idx_map: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    all_patches = []
    for window in range(1, 9):
        feat = FG.load_inference_features(window, grid)
        all_patches.append(FG.candidate_center_for_block(feat, means, store, cube, cube_idx_map, block, window))
    patch = pd.concat(all_patches, ignore_index=True)
    target = (
        base["type"].eq("grid")
        & base["region"].eq(FG.REGION)
        & base["horizon"].eq(TARGET_HORIZON)
        & base["level"].isin(FG.target_levels(TARGET_GROUP))
    )
    merged = ordered_lookup(base, target, patch)
    before_center = pd.to_numeric(merged["dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
    candidate_center = merged["center"].to_numpy(dtype="float64") % 360.0
    delta = FG.circ_abs_diff(candidate_center, before_center)
    selected = np.isfinite(delta) & (delta <= ROW_CENTER_SHIFT_MAX)
    selected_fraction = float(np.mean(selected)) if len(selected) else 0.0
    if selected_fraction < MIN_SELECTED_FRACTION:
        raise RuntimeError(
            f"selected fraction {selected_fraction:.3f} below {MIN_SELECTED_FRACTION:.3f}; refusing tiny patch"
        )

    lo_width, hi_width = base_half_widths(merged)
    idx = merged.loc[selected, "index"].to_numpy(dtype="int64")
    center = candidate_center[selected]
    patched = base.copy()
    patched.loc[idx, "dir_50"] = center
    patched.loc[idx, "dir_05"] = (center - lo_width[selected]) % 360.0
    patched.loc[idx, "dir_95"] = (center + hi_width[selected]) % 360.0

    stability = [
        {
            "block": f"{TARGET_GROUP}_d{TARGET_HORIZON}_{block.candidate}_selective_delta_le_{ROW_CENTER_SHIFT_MAX:g}",
            "total_target_rows": int(len(merged)),
            "selected_rows": int(selected.sum()),
            "selected_fraction": selected_fraction,
            "all_delta_mean": float(np.nanmean(delta)),
            "all_delta_p50": float(np.nanquantile(delta, 0.50)),
            "all_delta_p90": float(np.nanquantile(delta, 0.90)),
            "all_delta_p99": float(np.nanquantile(delta, 0.99)),
            "selected_delta_mean": float(np.nanmean(delta[selected])),
            "selected_delta_p50": float(np.nanquantile(delta[selected], 0.50)),
            "selected_delta_p90": float(np.nanquantile(delta[selected], 0.90)),
            "selected_delta_p99": float(np.nanquantile(delta[selected], 0.99)),
        }
    ]
    patch_counts = {stability[0]["block"]: int(selected.sum())}
    return patched, patch_counts, stability


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int, circular: bool) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if circular:
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def validate_delta(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    speed_changed = rows_changed(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, after, DIR_COLS, 1, circular=True)
    allowed = (
        after["type"].eq("grid")
        & after["region"].eq(FG.REGION)
        & after["horizon"].eq(TARGET_HORIZON)
        & after["level"].isin(FG.target_levels(TARGET_GROUP))
    ).to_numpy(dtype=bool)
    outside = dir_changed & ~allowed
    if int(speed_changed.sum()) != 0:
        raise SystemExit(f"unexpected speed rows changed: {int(speed_changed.sum())}")
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected non-target direction rows changed: {int(outside.sum())}")
    return {
        "allowed_target_rows": int(allowed.sum()),
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_direction_rows_changed": int(outside.sum()),
    }


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
    reason: str,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    block: FG.SelectedBlock | None,
    stability: list[dict[str, object]] | None = None,
    delta: dict[str, object] | None = None,
    patch_counts: dict[str, object] | None = None,
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "submission": zip_payload() if OUT_ZIP.exists() else None,
        "base_csv": {
            "path": str(BASE_CSV),
            "size": int(BASE_CSV.stat().st_size) if BASE_CSV.exists() else None,
            "sha256": sha256(BASE_CSV) if BASE_CSV.exists() else None,
        },
        "accepted_block": None if block is None else block.__dict__,
        "patch_counts": patch_counts or {},
        "inference_stability": stability or [],
        "delta": delta or {},
        "cv_by_fold_csv": str(CV_BY_FOLD),
        "cv_summary_csv": str(CV_SUMMARY),
        "decision_csv": str(DECISIONS_CSV),
        "cv_summary": summary.to_dict(orient="records"),
        "tight_decisions": decisions.to_dict(orient="records"),
        "gates": {
            "target_group": TARGET_GROUP,
            "target_horizon": TARGET_HORIZON,
            "row_center_shift_max": ROW_CENTER_SHIFT_MAX,
            "min_selected_fraction": MIN_SELECTED_FRACTION,
            "width_policy": "preserve_existing_interval_widths_per_row",
        },
        "compliance": FG.compliance_payload(),
        "code_hashes": {
            "build_ns_surface_d14_selective_center_candidate.py": sha256(Path(__file__).resolve()),
            "build_ns_grid_tight_center_blend_candidate.py": sha256(ROOT / "build_ns_grid_tight_center_blend_candidate.py"),
            "build_ns_grid_dir_regime_fast_gate_candidate.py": sha256(ROOT / "build_ns_grid_dir_regime_fast_gate_candidate.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(payload, indent=2, sort_keys=True))
    log(f"Wrote {MANIFEST}")


def main() -> None:
    require(BASE_CSV, "Run .\\run_ns_p7dir_mosres_e2e.ps1 first.")
    cleanup_outputs()
    configure_factories()

    grid, means, store, cube, cube_idx_map, summary = run_cv()
    block, decisions = select_surface_block(summary)
    if block is None:
        write_manifest(
            "gate_failed_no_submission_written",
            "No NS surface d14 center-blend candidate cleared the tight CV/public-margin gates.",
            summary,
            decisions,
            None,
        )
        return

    log(f"Selected selective block: {block.__dict__}")
    base = FG.normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    before = base.copy()
    try:
        patched, patch_counts, stability = apply_selective_center(base, block, grid, means, store, cube, cube_idx_map)
    except RuntimeError as exc:
        write_manifest(
            "gate_failed_no_submission_written",
            str(exc),
            summary,
            decisions,
            block,
        )
        return

    final = E2E.validate_final(patched)
    delta = validate_delta(before, final)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(
        "submission_written_after_selective_center_gates",
        "All gates passed.",
        summary,
        decisions,
        block,
        stability=stability,
        delta=delta,
        patch_counts=patch_counts,
    )
    log(f"OK: {OUT_ZIP}")


if __name__ == "__main__":
    main()
