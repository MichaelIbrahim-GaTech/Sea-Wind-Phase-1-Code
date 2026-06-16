from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E
import seasonal_direction_backtest as S
import build_seasonal_backtest_gate_candidate as B


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"
FEATURES = DATA / "features"

ORIG_BASE_CSV = WORK / "pred_sfc14_selw.csv"
CONFIRMED_BASE_CSV = WORK / "pred_nspos_b75.csv"
OUT_CSV = WORK / "pred_ns14_rowgate.csv"
OUT_ZIP = WORK / "sub_ns14_rowgate.zip"
MANIFEST = WORK / "manifest_ns14_rowgate.json"

HOURS = (0, 6, 12, 18)
TARGETS = (
    # Existing confirmed base already uses blend_seasonal_w21_0.75 for these.
    ("north_sea", "pressure", 14),
    ("north_sea", "surface", 14),
)
WINDOW_DAYS = 21
BASE_WEIGHT_CURRENT = 0.75
STRONG_WEIGHTS_CURRENT = (None, 0.25, 0.50)
DELTA_THRESHOLDS = (15.0, 20.0, 30.0, 45.0, 60.0, 90.0)
MIN_PROXY_GAIN = 1.0
MAX_CHANGED_FRACTION = 0.70
MAX_INFERENCE_SHIFT_P90 = 65.0
SAMPLE_PER_ANCHOR_DATE = 650


COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS


@dataclass(frozen=True)
class CandidateSpec:
    region: str
    group: str
    horizon: int
    weight_current: float | None
    delta_threshold: float
    proxy_score: float
    proxy_half_width: float
    proxy_gain: float
    changed_fraction: float


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
    for path in (OUT_CSV, OUT_ZIP, MANIFEST):
        if path.exists():
            path.unlink()


def require(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    out = E2E.normalize_for_assembly(df[COLS])
    out["type"] = out["type"].fillna("").astype(str).str.lower().str.strip()
    out["region"] = out["region"].fillna("").astype(str)
    out["station"] = out["station"].fillna("").astype(str)
    out["level"] = out["level"].fillna("").astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(2)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(2)
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    for c in SPEED_COLS + DIR_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def assert_same_order(left: pd.DataFrame, right: pd.DataFrame) -> None:
    keys = ["type", "window", "region", "latitude", "longitude", "station", "horizon", "hour", "level"]
    bad = False
    for c in keys:
        if not left[c].reset_index(drop=True).equals(right[c].reset_index(drop=True)):
            bad = True
            break
    if bad or len(left) != len(right):
        raise SystemExit("Original and confirmed base rows are not aligned; refusing row-index patching.")


def candidate_name(weight_current: float | None, threshold: float) -> str:
    if weight_current is None:
        w = "seasonal"
    else:
        w = f"blend{weight_current:.2f}"
    return f"{w}_dlt{threshold:g}"


def parse_target(raw: str) -> tuple[str, str, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be region,group,horizon, e.g. north_sea,surface,14")
    region, group, horizon = parts
    if group not in B.GROUP_LEVELS:
        raise argparse.ArgumentTypeError(f"unsupported group: {group}")
    return region, group, int(horizon)


def candidate_center(seasonal: np.ndarray, current: np.ndarray, weight_current: float | None) -> np.ndarray:
    if weight_current is None:
        return np.asarray(seasonal, dtype="float64") % 360.0
    return B.blend_direction(seasonal, current, float(weight_current)) % 360.0


def target_levels(group: str) -> tuple[str, ...]:
    return B.GROUP_LEVELS[group]


def load_model_centers(path: Path) -> pd.DataFrame:
    cols = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level", "dir_50"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = df[df["type"].astype(str).str.lower().eq("grid")].copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(2)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(2)
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("int64")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("int64")
    df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("int64")
    df["level"] = df["level"].astype(str)
    df["dir_50"] = pd.to_numeric(df["dir_50"], errors="coerce") % 360.0
    return df


def sampled_anchor_rows(region: str) -> pd.DataFrame:
    feat = pd.read_parquet(FEATURES / f"train_{region}.parquet", columns=["time", "latitude", "longitude"])
    feat["time"] = pd.to_datetime(feat["time"])
    feat["latitude"] = feat["latitude"].astype(float).round(2)
    feat["longitude"] = feat["longitude"].astype(float).round(2)
    eval_df = feat[feat["time"].isin(S.ANCHORS_2021)].copy()
    if SAMPLE_PER_ANCHOR_DATE > 0:
        eval_df = pd.concat(
            [
                p.sample(min(len(p), SAMPLE_PER_ANCHOR_DATE), random_state=2037)
                for _, p in eval_df.groupby("time", sort=True)
            ],
            ignore_index=True,
        )
    window_map = {d: i + 1 for i, d in enumerate(S.ANCHORS_2021)}
    eval_df["window"] = eval_df["time"].map(window_map).astype("int64")
    return eval_df


def block_proxy_arrays(
    region: str,
    group: str,
    horizon: int,
    orig_pred: pd.DataFrame,
    confirmed_pred: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual = S.prepare_surface_actual(region) if group == "surface" else S.prepare_pressure_actual(region)
    actual_hist = actual[actual["year"].isin([2019, 2020])].copy()
    actual_hist_by_hour = {
        int(hour): part.reset_index(drop=True)
        for hour, part in actual_hist.groupby("hour", sort=False)
    }
    eval_df = sampled_anchor_rows(region)
    coord_cols = ["latitude", "longitude"]

    y_parts: list[np.ndarray] = []
    orig_parts: list[np.ndarray] = []
    confirmed_parts: list[np.ndarray] = []
    seasonal_parts: list[np.ndarray] = []
    for anchor in S.ANCHORS_2021:
        anchor_rows = eval_df[eval_df["time"].eq(anchor)].copy()
        window_id = int(anchor_rows["window"].iloc[0])
        for hour in HOURS:
            target_time = anchor + pd.Timedelta(days=int(horizon)) + pd.Timedelta(hours=int(hour))
            for level in target_levels(group):
                level_col = f"dir_{level}"
                y = S.target_from_actual(actual, level_col, target_time, anchor_rows[coord_cols])
                y_parts.append(y)

                filt = (
                    orig_pred["region"].eq(region)
                    & orig_pred["window"].eq(window_id)
                    & orig_pred["horizon"].eq(horizon)
                    & orig_pred["hour"].eq(hour)
                    & orig_pred["level"].eq(level)
                )
                mp = orig_pred.loc[filt, ["latitude", "longitude", "dir_50"]]
                merged = anchor_rows[coord_cols].merge(mp, on=["latitude", "longitude"], how="left")
                orig_parts.append(merged["dir_50"].to_numpy(dtype="float64") % 360.0)

                filt = (
                    confirmed_pred["region"].eq(region)
                    & confirmed_pred["window"].eq(window_id)
                    & confirmed_pred["horizon"].eq(horizon)
                    & confirmed_pred["hour"].eq(hour)
                    & confirmed_pred["level"].eq(level)
                )
                mp = confirmed_pred.loc[filt, ["latitude", "longitude", "dir_50"]]
                merged = anchor_rows[coord_cols].merge(mp, on=["latitude", "longitude"], how="left")
                confirmed_parts.append(merged["dir_50"].to_numpy(dtype="float64") % 360.0)

                seas = S.seasonal_centers_for_windows(
                    actual_hist_by_hour,
                    level_col,
                    target_time,
                    anchor_rows[coord_cols],
                )[WINDOW_DAYS]
                seasonal_parts.append(seas)

    return (
        np.concatenate(y_parts),
        np.concatenate(orig_parts),
        np.concatenate(confirmed_parts),
        np.concatenate(seasonal_parts),
    )


def score_direction(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(y) & np.isfinite(pred)
    best = S.SOL.optimize_dir_halfwidth(y[good], pred[good], S.SOL.CFG.dir_halfwidth_grid)
    return float(best["score"]), float(best["half_width"])


def select_candidates(orig_pred: pd.DataFrame, confirmed_pred: pd.DataFrame) -> tuple[pd.DataFrame, list[CandidateSpec]]:
    rows: list[dict[str, object]] = []
    selected: list[CandidateSpec] = []
    for region, group, horizon in TARGETS:
        log(f"[proxy] Evaluating row-gated seasonal variants for {region}/{group}/d{horizon}")
        y, orig, confirmed, seasonal = block_proxy_arrays(region, group, horizon, orig_pred, confirmed_pred)
        base_score, base_width = score_direction(y, confirmed)
        seasonal_delta = B.circ_abs_diff(seasonal, orig)
        for weight_current in STRONG_WEIGHTS_CURRENT:
            strong = candidate_center(seasonal, orig, weight_current)
            for threshold in DELTA_THRESHOLDS:
                mask = seasonal_delta <= float(threshold)
                pred = confirmed.copy()
                pred[mask] = strong[mask]
                score, width = score_direction(y, pred)
                changed_fraction = float(np.mean(mask & np.isfinite(strong) & np.isfinite(confirmed)))
                gain = float(base_score - score)
                shift = B.circ_abs_diff(pred, confirmed)
                rows.append(
                    {
                        "region": region,
                        "group": group,
                        "horizon": horizon,
                        "base_score": base_score,
                        "base_half_width": base_width,
                        "candidate": candidate_name(weight_current, threshold),
                        "weight_current": "seasonal" if weight_current is None else float(weight_current),
                        "delta_threshold": float(threshold),
                        "score": score,
                        "half_width": width,
                        "gain_vs_confirmed": gain,
                        "changed_fraction": changed_fraction,
                        "shift_mean": float(np.nanmean(shift)),
                        "shift_p90": float(np.nanquantile(shift, 0.90)),
                    }
                )

        block_table = pd.DataFrame([r for r in rows if r["region"] == region and r["group"] == group and r["horizon"] == horizon])
        viable = block_table[
            (block_table["gain_vs_confirmed"] >= MIN_PROXY_GAIN)
            & (block_table["changed_fraction"] <= MAX_CHANGED_FRACTION)
        ].sort_values(["gain_vs_confirmed", "score"], ascending=[False, True], kind="mergesort")
        if viable.empty:
            log(f"[proxy] No viable row-gated seasonal variant for {region}/{group}/d{horizon}")
            continue
        best = viable.iloc[0]
        weight = None if best["weight_current"] == "seasonal" else float(best["weight_current"])
        selected.append(
            CandidateSpec(
                region=region,
                group=group,
                horizon=int(horizon),
                weight_current=weight,
                delta_threshold=float(best["delta_threshold"]),
                proxy_score=float(best["score"]),
                proxy_half_width=float(best["half_width"]),
                proxy_gain=float(best["gain_vs_confirmed"]),
                changed_fraction=float(best["changed_fraction"]),
            )
        )
        log(
            "[proxy] selected "
            f"{region}/{group}/d{horizon}: {best['candidate']} "
            f"gain={float(best['gain_vs_confirmed']):.3f} "
            f"score={float(best['score']):.3f} width={float(best['half_width']):.1f}"
        )
    return pd.DataFrame(rows), selected


def window_metadata(window: int) -> dict[str, object]:
    return json.loads((DATA / "inference" / f"window_{window}" / "metadata.json").read_text(encoding="utf-8"))


def apply_candidates(
    orig: pd.DataFrame,
    confirmed: pd.DataFrame,
    selected: list[CandidateSpec],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    patched = confirmed.copy()
    actual_cache: dict[tuple[str, str], pd.DataFrame] = {}
    hist_cache: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
    stability: list[dict[str, object]] = []

    for spec in selected:
        actual = actual_cache.setdefault((spec.region, spec.group), B.load_actual(spec.region, spec.group))
        hist_by_hour = hist_cache.setdefault(
            (spec.region, spec.group),
            {int(hour): part.reset_index(drop=True) for hour, part in actual[actual["year"].le(2021)].groupby("hour", sort=False)},
        )
        changed = 0
        target_rows = 0
        shift_parts: list[np.ndarray] = []
        seasonal_delta_parts: list[np.ndarray] = []
        log(
            f"[apply] {spec.region}/{spec.group}/d{spec.horizon}: "
            f"{candidate_name(spec.weight_current, spec.delta_threshold)}, width={spec.proxy_half_width:.1f}"
        )
        half_width = float(np.clip(round(spec.proxy_half_width / 5.0) * 5.0, 15.0, 179.9))
        for window in range(1, 9):
            meta = window_metadata(window)
            score_day = pd.Timestamp(meta["score_days"][f"d{spec.horizon}"])
            for hour in HOURS:
                target_time = score_day + pd.Timedelta(hours=int(hour))
                for level in target_levels(spec.group):
                    idx = patched.index[
                        patched["type"].eq("grid")
                        & patched["region"].eq(spec.region)
                        & patched["window"].eq(window)
                        & patched["horizon"].eq(spec.horizon)
                        & patched["hour"].eq(hour)
                        & patched["level"].eq(level)
                    ]
                    if len(idx) == 0:
                        raise SystemExit(f"missing rows for {spec.region}/{spec.group}/d{spec.horizon}/w{window}/h{hour}/{level}")
                    coords = patched.loc[idx, ["latitude", "longitude"]].copy()
                    seasonal = B.seasonal_centers(hist_by_hour, level, target_time, coords, WINDOW_DAYS)
                    if np.isnan(seasonal).any():
                        raise SystemExit(f"seasonal center has missing values for {spec.region}/{spec.group}/d{spec.horizon}/w{window}/h{hour}/{level}")
                    orig_center = pd.to_numeric(orig.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
                    base_center = pd.to_numeric(patched.loc[idx, "dir_50"], errors="coerce").to_numpy(dtype="float64") % 360.0
                    strong = candidate_center(seasonal, orig_center, spec.weight_current)
                    seasonal_delta = B.circ_abs_diff(seasonal, orig_center)
                    mask = seasonal_delta <= spec.delta_threshold
                    if np.any(mask):
                        chosen = strong[mask] % 360.0
                        patch_idx = idx[mask]
                        patched.loc[patch_idx, "dir_50"] = chosen
                        patched.loc[patch_idx, "dir_05"] = (chosen - half_width) % 360.0
                        patched.loc[patch_idx, "dir_95"] = (chosen + half_width) % 360.0
                        shift_parts.append(B.circ_abs_diff(chosen, base_center[mask]))
                        seasonal_delta_parts.append(seasonal_delta[mask])
                        changed += int(len(patch_idx))
                    target_rows += int(len(idx))
        shifts = np.concatenate(shift_parts) if shift_parts else np.array([], dtype="float64")
        seas_delta = np.concatenate(seasonal_delta_parts) if seasonal_delta_parts else np.array([], dtype="float64")
        entry = {
            "block": f"{spec.region}_{spec.group}_d{spec.horizon}_{candidate_name(spec.weight_current, spec.delta_threshold)}",
            "target_rows": int(target_rows),
            "changed_rows": int(changed),
            "changed_fraction": float(changed / max(target_rows, 1)),
            "half_width": half_width,
            "proxy_score": spec.proxy_score,
            "proxy_gain": spec.proxy_gain,
            "inference_shift_mean": float(np.nanmean(shifts)) if len(shifts) else 0.0,
            "inference_shift_p90": float(np.nanquantile(shifts, 0.90)) if len(shifts) else 0.0,
            "inference_shift_p99": float(np.nanquantile(shifts, 0.99)) if len(shifts) else 0.0,
            "seasonal_current_delta_p90_changed": float(np.nanquantile(seas_delta, 0.90)) if len(seas_delta) else 0.0,
        }
        stability.append(entry)
        if entry["inference_shift_p90"] > MAX_INFERENCE_SHIFT_P90:
            raise SystemExit(
                f"row-gated candidate failed inference-shift gate: {entry['block']} "
                f"p90={entry['inference_shift_p90']:.2f}"
            )
    return patched, stability


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


def validate_delta(before: pd.DataFrame, after: pd.DataFrame, selected: list[CandidateSpec]) -> dict[str, object]:
    speed_changed = rows_changed(before, after, SPEED_COLS, 2, circular=False)
    dir_changed = rows_changed(before, after, DIR_COLS, 1, circular=True)
    allowed = np.zeros(len(after), dtype=bool)
    target_rows_by_block: dict[str, int] = {}
    for spec in selected:
        mask = (
            after["type"].eq("grid")
            & after["region"].eq(spec.region)
            & after["horizon"].eq(spec.horizon)
            & after["level"].isin(target_levels(spec.group))
        ).to_numpy(dtype=bool)
        allowed |= mask
        target_rows_by_block[f"{spec.region}_{spec.group}_d{spec.horizon}"] = int(mask.sum())
    outside = dir_changed & ~allowed
    if int(speed_changed.sum()) != 0:
        raise SystemExit(f"unexpected speed rows changed: {int(speed_changed.sum())}")
    if int(outside.sum()) != 0:
        raise SystemExit(f"unexpected direction rows outside selected blocks: {int(outside.sum())}")
    return {
        "target_rows_by_block": target_rows_by_block,
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
    proxy_table: pd.DataFrame,
    selected: list[CandidateSpec],
    stability: list[dict[str, object]] | None = None,
    delta: dict[str, object] | None = None,
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "submission": zip_payload() if OUT_ZIP.exists() else None,
        "original_base_csv": {
            "path": str(ORIG_BASE_CSV),
            "size": int(ORIG_BASE_CSV.stat().st_size) if ORIG_BASE_CSV.exists() else None,
            "sha256": sha256(ORIG_BASE_CSV) if ORIG_BASE_CSV.exists() else None,
        },
        "confirmed_base_csv": {
            "path": str(CONFIRMED_BASE_CSV),
            "size": int(CONFIRMED_BASE_CSV.stat().st_size) if CONFIRMED_BASE_CSV.exists() else None,
            "sha256": sha256(CONFIRMED_BASE_CSV) if CONFIRMED_BASE_CSV.exists() else None,
        },
        "targets": [{"region": r, "group": g, "horizon": h} for r, g, h in TARGETS],
        "candidate_grid": {
            "seasonal_window_days": WINDOW_DAYS,
            "base_weight_current": BASE_WEIGHT_CURRENT,
            "strong_weights_current": ["seasonal" if w is None else w for w in STRONG_WEIGHTS_CURRENT],
            "delta_thresholds": list(DELTA_THRESHOLDS),
            "min_proxy_gain": MIN_PROXY_GAIN,
            "max_changed_fraction": MAX_CHANGED_FRACTION,
            "max_inference_shift_p90": MAX_INFERENCE_SHIFT_P90,
            "sample_per_anchor_date": SAMPLE_PER_ANCHOR_DATE,
        },
        "selected": [
            {
                "region": s.region,
                "group": s.group,
                "horizon": s.horizon,
                "weight_current": "seasonal" if s.weight_current is None else s.weight_current,
                "delta_threshold": s.delta_threshold,
                "proxy_score": s.proxy_score,
                "proxy_half_width": s.proxy_half_width,
                "proxy_gain": s.proxy_gain,
                "changed_fraction": s.changed_fraction,
            }
            for s in selected
        ],
        "proxy_table_top": proxy_table.sort_values(
            ["gain_vs_confirmed", "score"], ascending=[False, True], kind="mergesort"
        ).head(80).to_dict(orient="records")
        if not proxy_table.empty
        else [],
        "inference_stability": stability or [],
        "delta": delta or {},
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "future_target_data_used": False,
            "notes": [
                "Starts from generated official-data bases pred_sfc14_selw.csv and confirmed pred_nspos_b75.csv.",
                "Proxy selection uses only official 2019-2021 train/reanalysis data and generated predictions.",
                "Final inference seasonal centers use official 2019-2021 train reanalysis, all before 2022 inference windows.",
                "Public leaderboard metrics are not used as model features or labels.",
            ],
        },
        "code_hashes": {
            "build_seasonal_rowgate_candidate.py": sha256(Path(__file__).resolve()),
            "seasonal_direction_backtest.py": sha256(ROOT / "seasonal_direction_backtest.py"),
            "build_seasonal_backtest_gate_candidate.py": sha256(ROOT / "build_seasonal_backtest_gate_candidate.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(payload, indent=2, sort_keys=True))
    log(f"Wrote {MANIFEST}")


def main() -> None:
    global ORIG_BASE_CSV, CONFIRMED_BASE_CSV, OUT_CSV, OUT_ZIP, MANIFEST, TARGETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-base-csv", type=Path, default=ORIG_BASE_CSV)
    parser.add_argument("--confirmed-base-csv", type=Path, default=CONFIRMED_BASE_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-zip", type=Path, default=OUT_ZIP)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        help="Optional target block as region,group,horizon. May be repeated. Defaults to NS pressure d14 and NS surface d14.",
    )
    args = parser.parse_args()
    ORIG_BASE_CSV = args.orig_base_csv
    CONFIRMED_BASE_CSV = args.confirmed_base_csv
    OUT_CSV = args.out_csv
    OUT_ZIP = args.out_zip
    MANIFEST = args.manifest
    if args.target:
        TARGETS = tuple(args.target)

    require(ORIG_BASE_CSV, "Run the base generator first.")
    require(CONFIRMED_BASE_CSV, "Run run_seasonal_ns_positive_e2e.ps1 first.")
    cleanup_outputs()

    orig_pred = load_model_centers(ORIG_BASE_CSV)
    confirmed_pred = load_model_centers(CONFIRMED_BASE_CSV)
    proxy_table, selected = select_candidates(orig_pred, confirmed_pred)
    if not selected:
        write_manifest(
            "gate_failed_no_submission_written",
            "No row-gated seasonal variant improved the confirmed base under the strict proxy gates.",
            proxy_table,
            selected,
        )
        return

    orig = normalize_base(pd.read_csv(ORIG_BASE_CSV, low_memory=False))
    confirmed = normalize_base(pd.read_csv(CONFIRMED_BASE_CSV, low_memory=False))
    assert_same_order(orig, confirmed)
    before = confirmed.copy()
    patched, stability = apply_candidates(orig, confirmed, selected)
    final = E2E.validate_final(patched)
    delta = validate_delta(before, final, selected)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(
        "submission_written_after_rowgate_proxy_and_audit",
        "Row-gated seasonal strengthening passed proxy, inference-shift, schema, delta, and zip gates.",
        proxy_table,
        selected,
        stability=stability,
        delta=delta,
    )
    log(f"OK: {OUT_ZIP}")


if __name__ == "__main__":
    main()
