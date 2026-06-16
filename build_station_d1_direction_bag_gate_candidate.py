from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sea_winds_end_to_end_final as E2E
import station_cv_mos_analog_framework as CV
from build_ns_d1dir_enswidth_gate_candidate import (
    apply_angle_bias,
    build_angle_bias_map,
    circ_mean_matrix,
    direction_score_var,
    fit_width_policy,
    fit_xy_models,
    lookup_angle_bias,
    predict_width,
    predict_xy_direction,
)
from build_station_lgbm_ns_d1_speed_on_analog_candidate import (
    load_inference_origin_rows,
    load_station_obs_with_context,
)


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

BASE_CSV = WORK / "pred_ns_d1spd_calib_gate.csv"
OUT_CSV = WORK / "pred_stndir_d1_bag_gate.csv"
OUT_ZIP = WORK / "sub_stndir_d1_bag_gate.zip"
SUMMARY_CSV = WORK / "cv_stndir_d1_bag_gate.csv"
MANIFEST = WORK / "manifest_stndir_d1_bag_gate.json"

HORIZON = 1
HOURS = CV.HOURS
COLS = E2E.COLS
SPEED_COLS = E2E.SPEED_COLS
DIR_COLS = E2E.DIR_COLS

SEED_BASE = {"north_sea": 20260602, "east_china_sea": 20261602}
SEED_OFFSETS = (0, 1)
FAMILIES = ("direct_unit", "unit_residual")

MEMBERS = {
    "du_s8": {"family": "direct_unit", "group_cols": ["station"], "shrink": 8.0},
    "du_s20": {"family": "direct_unit", "group_cols": ["station"], "shrink": 20.0},
    "du_sh12": {"family": "direct_unit", "group_cols": ["station", "target_hour"], "shrink": 12.0},
    "ur_s8": {"family": "unit_residual", "group_cols": ["station"], "shrink": 8.0},
    "ur_sh12": {"family": "unit_residual", "group_cols": ["station", "target_hour"], "shrink": 12.0},
}

CENTER_SPECS = [
    {"name": "du_s8_bag", "members": ["du_s8"]},
    {"name": "du_sh12_bag", "members": ["du_sh12"]},
    {"name": "ur_s8_bag", "members": ["ur_s8"]},
    {"name": "ens_du_s8_sh12_ur8", "members": ["du_s8", "du_sh12", "ur_s8"]},
    {"name": "ens_du_s8_s20_ur8", "members": ["du_s8", "du_s20", "ur_s8"]},
    {"name": "ens_du_sh12_ur_sh12", "members": ["du_sh12", "ur_sh12"]},
]

WIDTH_POLICIES = [
    {"name": "fixed52_5", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 52.5},
    {"name": "fixed55", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 55.0},
    {"name": "fixed57_5", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 57.5},
    {"name": "fixed60", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 60.0},
    {"name": "fixed65", "kind": "fixed", "group_cols": [], "shrink": 0.0, "value": 65.0},
    {"name": "station_safe_s40", "kind": "adaptive", "group_cols": ["station"], "shrink": 40.0, "safe_grid": True},
    {"name": "hour_safe_s16", "kind": "adaptive", "group_cols": ["target_hour"], "shrink": 16.0, "safe_grid": True},
    {"name": "station_hour_safe_s80", "kind": "adaptive", "group_cols": ["station", "target_hour"], "shrink": 80.0, "safe_grid": True},
]

REGION_GATES = {
    "north_sea": {
        "label": "Dir NS Stations d1",
        "expected_rows": 256,
        "baseline_cv_mean": 188.4860510719073,
        "baseline_cv_max": 207.59321683951535,
        "required_mean_improvement": 0.20,
        "max_margin": 1.00,
        "public_current": 185.6761,
    },
    "east_china_sea": {
        "label": "Dir ECS Stations d1",
        "expected_rows": 224,
        "baseline_cv_mean": 181.76949201472724,
        "baseline_cv_max": 185.64698587918292,
        "required_mean_improvement": 0.20,
        "max_margin": 1.50,
        "public_current": 230.5785,
    },
}


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
    for c in ["window", "horizon", "hour"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    return out


def load_station_training(region: str, meta: pd.DataFrame) -> pd.DataFrame:
    train_base = CV.load_station_origin_rows(region, meta)
    hist = CV.make_history(CV.load_station_obs(region))
    df = CV.build_combo(train_base, hist, HORIZON, include_climatology=False)
    df = df[df["y_dir"].notna() & df["hres_dir"].notna() & df["hres_speed"].notna()].copy()
    return df.reset_index(drop=True)


def fit_raw_models(train: pd.DataFrame, feats: list[str], region: str, val_year: int | None) -> dict[tuple[str, int], tuple[object, object]]:
    models: dict[tuple[str, int], tuple[object, object]] = {}
    base = SEED_BASE[region] + (9999 if val_year is None else int(val_year) * 100)
    for f_idx, family in enumerate(FAMILIES):
        for s_idx, offset in enumerate(SEED_OFFSETS):
            seed = base + f_idx * 31 + offset * 997
            print(f"  fitting {region} {family} seed{offset} seed={seed}", flush=True)
            models[(family, s_idx)] = fit_xy_models(train, feats, family, seed)
    return models


def raw_predictions(models: dict[tuple[str, int], tuple[object, object]], feats: list[str], df: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {key: predict_xy_direction(model, feats, df, key[0]) for key, model in models.items()}


def fit_member_biases(
    member_name: str,
    train: pd.DataFrame,
    raw_train: dict[tuple[str, int], np.ndarray],
) -> dict[str, object]:
    spec = MEMBERS[member_name]
    family = str(spec["family"])
    group_cols = list(spec["group_cols"])
    shrink = float(spec["shrink"])
    seed_fits = []
    for s_idx, _offset in enumerate(SEED_OFFSETS):
        bias_map, global_bias = build_angle_bias_map(train, raw_train[(family, s_idx)], group_cols, shrink)
        seed_fits.append({"seed_idx": s_idx, "bias_map": bias_map, "global_bias": float(global_bias)})
    return {"spec": spec, "seed_fits": seed_fits}


def apply_member(
    member_name: str,
    df: pd.DataFrame,
    raw_pred: dict[tuple[str, int], np.ndarray],
    member_fit: dict[str, object],
) -> np.ndarray:
    spec = MEMBERS[member_name]
    family = str(spec["family"])
    group_cols = list(spec["group_cols"])
    preds = []
    for seed_fit in member_fit["seed_fits"]:
        s_idx = int(seed_fit["seed_idx"])
        bias = lookup_angle_bias(df, seed_fit["bias_map"], float(seed_fit["global_bias"]), group_cols)
        preds.append(apply_angle_bias(raw_pred[(family, s_idx)], bias))
    return circ_mean_matrix(preds)


def fit_member_set(train: pd.DataFrame, raw_train: dict[tuple[str, int], np.ndarray]) -> dict[str, dict[str, object]]:
    needed = sorted({name for spec in CENTER_SPECS for name in spec["members"]})
    return {name: fit_member_biases(name, train, raw_train) for name in needed}


def build_centers(
    pred_df: pd.DataFrame,
    raw_pred: dict[tuple[str, int], np.ndarray],
    member_fits: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    member_preds = {name: apply_member(name, pred_df, raw_pred, fit) for name, fit in member_fits.items()}
    centers = {}
    for spec in CENTER_SPECS:
        centers[spec["name"]] = circ_mean_matrix([member_preds[name] for name in spec["members"]])
    return centers


def evaluate_region(region: str, df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows = []
    for val_year in (2020, 2021):
        train = df[df["time"].dt.year.lt(val_year)].copy()
        val = df[CV.anchor_mask(df, val_year)].copy()
        feats = CV.numeric_features(train)
        print(f"CV {region} fold {val_year}: train={len(train):,} val={len(val):,} features={len(feats)}", flush=True)
        models = fit_raw_models(train, feats, region, val_year)
        raw_train = raw_predictions(models, feats, train)
        raw_val = raw_predictions(models, feats, val)
        member_fits = fit_member_set(train, raw_train)
        train_centers = build_centers(train, raw_train, member_fits)
        val_centers = build_centers(val, raw_val, member_fits)
        y_val = val["y_dir"].to_numpy(dtype="float64")
        for center_spec in CENTER_SPECS:
            center_name = center_spec["name"]
            for width_policy in WIDTH_POLICIES:
                width_fit = fit_width_policy(train, train_centers[center_name], width_policy)
                widths = predict_width(val, width_fit)
                rows.append(
                    {
                        "region": region,
                        "val_year": int(val_year),
                        "center": center_name,
                        "members": ",".join(center_spec["members"]),
                        "width_policy": width_policy["name"],
                        "width_kind": width_policy["kind"],
                        "width_group_cols": ",".join(width_policy["group_cols"]),
                        "width_shrink": float(width_policy["shrink"]),
                        "score": direction_score_var(y_val, val_centers[center_name], widths),
                        "half_width_mean": float(np.nanmean(widths)),
                        "half_width_min": float(np.nanmin(widths)),
                        "half_width_max": float(np.nanmax(widths)),
                        "train_rows": int(len(train)),
                        "val_rows": int(len(val)),
                    }
                )
    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby(["region", "center", "members", "width_policy", "width_kind", "width_group_cols", "width_shrink"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            half_width_mean=("half_width_mean", "mean"),
            half_width_min=("half_width_min", "min"),
            half_width_max=("half_width_max", "max"),
        )
        .sort_values(["score_mean", "score_max"], kind="mergesort")
        .reset_index(drop=True)
    )
    gate = REGION_GATES[region]
    summary["baseline_cv_mean"] = float(gate["baseline_cv_mean"])
    summary["baseline_cv_max"] = float(gate["baseline_cv_max"])
    summary["mean_improvement"] = summary["baseline_cv_mean"] - summary["score_mean"]
    summary["max_delta"] = summary["score_max"] - summary["baseline_cv_max"]
    summary["gate_passed"] = (
        (summary["score_mean"] <= float(gate["baseline_cv_mean"]) - float(gate["required_mean_improvement"]))
        & (summary["score_max"] <= float(gate["baseline_cv_max"]) + float(gate["max_margin"]))
    )
    print(f"{region} bag direction summary:", flush=True)
    print(summary.head(16).to_string(index=False), flush=True)
    passed = summary[summary["gate_passed"]].copy()
    if passed.empty:
        selected = summary.iloc[0].to_dict()
        selected["gate_passed"] = False
    else:
        selected = passed.sort_values(["score_mean", "score_max"], kind="mergesort").iloc[0].to_dict()
        selected["gate_passed"] = True
    selected["gate"] = gate
    print(f"Selected {region}: {selected}", flush=True)
    return selected, cv, summary


def fit_final_region(region: str, df: pd.DataFrame, selected: dict[str, object]) -> dict[str, object]:
    feats = CV.numeric_features(df)
    models = fit_raw_models(df, feats, region, val_year=None)
    raw_train = raw_predictions(models, feats, df)
    member_fits = fit_member_set(df, raw_train)
    train_centers = build_centers(df, raw_train, member_fits)
    policy = next(p for p in WIDTH_POLICIES if p["name"] == selected["width_policy"])
    width_fit = fit_width_policy(df, train_centers[str(selected["center"])], policy)
    return {"feats": feats, "models": models, "member_fits": member_fits, "width_fit": width_fit}


def make_region_patch(region: str, meta: pd.DataFrame, final_fit: dict[str, object], selected: dict[str, object]) -> pd.DataFrame:
    rows = []
    for window in range(1, 9):
        inf_base = load_inference_origin_rows(region, meta, window)
        inf_hist = CV.make_history(load_station_obs_with_context(region, window))
        for hour in HOURS:
            inf = CV.add_target_features(inf_base, inf_hist, HORIZON, hour, include_climatology=False)
            raw_pred = raw_predictions(final_fit["models"], final_fit["feats"], inf)
            centers = build_centers(inf, raw_pred, final_fit["member_fits"])
            center = centers[str(selected["center"])]
            widths = predict_width(inf, final_fit["width_fit"])
            for station, c, w in zip(inf["station"].astype(str), center, widths):
                rows.append(
                    {
                        "window": int(window),
                        "region": region,
                        "station": station,
                        "horizon": HORIZON,
                        "hour": int(hour),
                        "dir_05_new": float((c - w) % 360.0),
                        "dir_50_new": float(c % 360.0),
                        "dir_95_new": float((c + w) % 360.0),
                    }
                )
    return pd.DataFrame(rows)


def rows_changed(before: pd.DataFrame, after: pd.DataFrame, cols: list[str], decimals: int) -> np.ndarray:
    left = before[cols].apply(pd.to_numeric, errors="coerce")
    right = after[cols].apply(pd.to_numeric, errors="coerce")
    if set(cols) == set(DIR_COLS):
        left = (left % 360.0).round(decimals) % 360.0
        right = (right % 360.0).round(decimals) % 360.0
    else:
        left = left.round(decimals)
        right = right.round(decimals)
    return (left.to_numpy() != right.to_numpy()).any(axis=1)


def apply_patches(base: pd.DataFrame, patches: list[pd.DataFrame], selected_by_region: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    if not patches:
        raise SystemExit("No region passed the station d1 direction bag gate; no submission zip written.")
    patch = pd.concat(patches, ignore_index=True)
    merged = base.reset_index().merge(
        patch,
        on=["window", "region", "station", "horizon", "hour"],
        how="left",
        validate="many_to_one",
    )
    target = merged["type"].eq("station") & merged["horizon"].eq(HORIZON) & merged["dir_50_new"].notna()
    patch_counts: dict[str, int] = {}
    for region in selected_by_region:
        if selected_by_region[region].get("gate_passed"):
            count = int((target & merged["region"].eq(region)).sum())
            expected = int(REGION_GATES[region]["expected_rows"])
            if count != expected:
                raise SystemExit(f"{region}: expected {expected} patched rows, got {count}")
            patch_counts[region] = count
    before = base.copy()
    for c in DIR_COLS:
        merged.loc[target, c] = merged.loc[target, f"{c}_new"]
    out = merged.drop(columns=["dir_05_new", "dir_50_new", "dir_95_new"]).set_index("index").sort_index()[COLS]
    speed_changed = rows_changed(before, out, SPEED_COLS, 2)
    dir_changed = rows_changed(before, out, DIR_COLS, 1)
    changed_allowed = target.to_numpy(dtype=bool)
    outside_dir = dir_changed & ~changed_allowed
    if int(speed_changed.sum()) != 0 or int(outside_dir.sum()) != 0:
        raise SystemExit(f"Unexpected delta: speed_changed={int(speed_changed.sum())}, outside_dir={int(outside_dir.sum())}")
    delta = {
        "patched_counts": patch_counts,
        "target_rows": int(target.sum()),
        "speed_rows_changed": int(speed_changed.sum()),
        "direction_rows_changed": int(dir_changed.sum()),
        "non_target_direction_rows_changed": int(outside_dir.sum()),
    }
    return out, delta


def write_gate_failed_manifest(selected_by_region: dict[str, dict[str, object]], summaries: pd.DataFrame, cv: pd.DataFrame) -> None:
    payload = {
        "status": "gate_failed_no_submission_written",
        "selected_by_region": selected_by_region,
        "summary_head": summaries.groupby("region", group_keys=False).head(12).to_dict(orient="records"),
        "cv_rows": int(len(cv)),
        "compliance": [
            "Uses only official files under runs/v6_pressure_speed/phase1_dataset.",
            "No external datasets, no web data, and no evaluation target labels.",
            "Gate failed for all regions, so no submission zip was emitted.",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote gate-failed manifest {MANIFEST}", flush=True)


def write_manifest(final: pd.DataFrame, selected_by_region: dict[str, dict[str, object]], delta: dict[str, object], summaries: pd.DataFrame) -> None:
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        info = zf.getinfo("predictions.csv")
        bad = zf.testzip()
    if names != ["predictions.csv"] or bad is not None:
        raise SystemExit(f"bad zip: names={names}, testzip={bad}")
    manifest = {
        "submission": {
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
        },
        "base_csv": {"path": str(BASE_CSV), "size": int(BASE_CSV.stat().st_size), "sha256": sha256(BASE_CSV)},
        "selected_by_region": selected_by_region,
        "delta": delta,
        "cv": {
            "summary_csv": str(SUMMARY_CSV),
            "summary_sha256": sha256(SUMMARY_CSV),
            "summary_head": summaries.groupby("region", group_keys=False).head(12).to_dict(orient="records"),
        },
        "validation": {
            "rows": int(len(final)),
            "type_counts": {str(k): int(v) for k, v in final["type"].value_counts(dropna=False).to_dict().items()},
        },
        "compliance": {
            "official_dataset_root": str(DATA),
            "external_training_data_used": False,
            "web_data_used": False,
            "evaluation_target_labels_used_for_training": False,
            "notes": [
                "Training labels come only from official historical station observations.",
                "Inference history uses provided context station files for each official window.",
                "NS and ECS station d1 direction gates are independent; failed regions remain unchanged.",
                "The submission zip is emitted only if at least one region passes its chronological CV gate.",
            ],
        },
        "code_hashes": {
            "build_station_d1_direction_bag_gate_candidate.py": sha256(Path(__file__).resolve()),
            "build_ns_d1dir_enswidth_gate_candidate.py": sha256(ROOT / "build_ns_d1dir_enswidth_gate_candidate.py"),
            "build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py": sha256(ROOT / "build_station_uvres_ns_d1_direction_gate_on_stage2_candidate.py"),
            "station_cv_mos_analog_framework.py": sha256(ROOT / "station_cv_mos_analog_framework.py"),
            "sea_winds_end_to_end_final.py": sha256(ROOT / "sea_winds_end_to_end_final.py"),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {MANIFEST}", flush=True)


def main() -> None:
    require(BASE_CSV, "Run .\\run_ns_station_d1_speed_calib_gate_e2e.ps1 first.")
    print(f"Reading current best base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    base = normalize_base(pd.read_csv(BASE_CSV, low_memory=False))
    meta = CV.read_station_meta()

    selected_by_region: dict[str, dict[str, object]] = {}
    cv_parts = []
    summary_parts = []
    training_by_region = {}
    for region in ("north_sea", "east_china_sea"):
        df = load_station_training(region, meta)
        training_by_region[region] = df
        selected, cv, summary = evaluate_region(region, df)
        selected_by_region[region] = selected
        cv_parts.append(cv)
        summary_parts.append(summary)
    cv_all = pd.concat(cv_parts, ignore_index=True)
    summary_all = pd.concat(summary_parts, ignore_index=True)
    cv_all.merge(
        summary_all.add_prefix("summary_"),
        left_on=["region", "center", "members", "width_policy", "width_kind", "width_group_cols", "width_shrink"],
        right_on=[
            "summary_region",
            "summary_center",
            "summary_members",
            "summary_width_policy",
            "summary_width_kind",
            "summary_width_group_cols",
            "summary_width_shrink",
        ],
        how="left",
    ).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}", flush=True)

    patches = []
    for region, selected in selected_by_region.items():
        if bool(selected.get("gate_passed")):
            print(f"Fitting final gated model for {region}", flush=True)
            final_fit = fit_final_region(region, training_by_region[region], selected)
            patches.append(make_region_patch(region, meta, final_fit, selected))
        else:
            print(f"Leaving {region} unchanged; gate failed", flush=True)
    if not patches:
        write_gate_failed_manifest(selected_by_region, summary_all, cv_all)
        raise SystemExit("No region passed the station d1 direction bag gate; no submission zip written.")

    patched, delta = apply_patches(base, patches, selected_by_region)
    final = E2E.validate_final(patched)
    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    write_manifest(final, selected_by_region, delta, summary_all)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
