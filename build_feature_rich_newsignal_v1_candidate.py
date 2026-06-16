from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor

import build_learned_residual_newsignal_v1_candidate as LRN
import build_regime_newsignal_v1_candidate as RNS


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"
DATA = WORK / "phase1_dataset"

SAMPLE_PER_ANCHOR_DATE = 180
BASE_ROW_CACHE = WORK / f"regime_newsignal_v1_rows_s{SAMPLE_PER_ANCHOR_DATE}.parquet"
ENRICHED_ROW_CACHE = WORK / f"feature_rich_newsignal_v1_rows_s{SAMPLE_PER_ANCHOR_DATE}.parquet"

OUT_CSV = WORK / "pred_feature_rich_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_featrich_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_feature_rich_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_feature_rich_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_feature_rich_newsignal_v1.csv"
MANIFEST = WORK / "manifest_feature_rich_newsignal_v1.json"

KEY_COLS = ["time", "latitude", "longitude"]
EXCLUDE_FEATURE_COLS = {
    "time",
    "latitude",
    "longitude",
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def lgbm_model(seed: int, loss: str = "squared_error", quantile: float | None = None) -> LGBMRegressor:
    if loss == "quantile":
        objective = "quantile"
        alpha = 0.90 if quantile is None else float(quantile)
    else:
        objective = "regression"
        alpha = 0.90
    return LGBMRegressor(
        objective=objective,
        alpha=alpha,
        n_estimators=420,
        learning_rate=0.030,
        num_leaves=47,
        max_depth=8,
        min_child_samples=110,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=5.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def feature_path(region: str, inference_window: int | None = None) -> Path:
    if inference_window is None:
        return DATA / "features" / f"train_{region}.parquet"
    return DATA / "features" / f"inference_window_{inference_window}_{region}.parquet"


def allowed_feature_columns(region: str) -> list[str]:
    train_cols = set(pq.ParquetFile(feature_path(region)).schema_arrow.names)
    inference_cols = set(pq.ParquetFile(feature_path(region, 1)).schema_arrow.names)
    common = train_cols & inference_cols
    cols = sorted(c for c in common if c not in EXCLUDE_FEATURE_COLS)
    return cols


def add_merge_keys(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_merge_time"] = pd.to_datetime(out[time_col]).dt.normalize()
    out["_lat_key"] = np.rint(pd.to_numeric(out["latitude"], errors="coerce").to_numpy(dtype="float64") * 10000).astype("int64")
    out["_lon_key"] = np.rint(pd.to_numeric(out["longitude"], errors="coerce").to_numpy(dtype="float64") * 10000).astype("int64")
    return out


def load_region_features(region: str, needed_times: set[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    cols = allowed_feature_columns(region)
    read_cols = KEY_COLS + cols
    print(f"[features] loading {region} cols={len(cols)}", flush=True)
    feat = pd.read_parquet(feature_path(region), columns=read_cols)
    feat = add_merge_keys(feat, "time")
    if needed_times:
        feat = feat[feat["_merge_time"].isin(needed_times)].copy()
    rename = {c: f"ctx_{c}" for c in cols}
    feat = feat.rename(columns=rename)
    keep_cols = ["_merge_time", "_lat_key", "_lon_key"] + list(rename.values())
    feat = feat[keep_cols].drop_duplicates(["_merge_time", "_lat_key", "_lon_key"], keep="last")
    print(f"[features] {region} rows={len(feat):,}", flush=True)
    return feat, list(rename.values())


def ensure_base_row_cache() -> None:
    if BASE_ROW_CACHE.exists():
        print(f"[cache] using {BASE_ROW_CACHE}", flush=True)
        return
    print(f"[cache] building {BASE_ROW_CACHE}", flush=True)
    RNS.SAMPLE_PER_ANCHOR_DATE = SAMPLE_PER_ANCHOR_DATE
    RNS.ROW_CACHE = BASE_ROW_CACHE
    RNS.install_fast_anchor_predictors()
    rows = pd.concat([RNS.build_direction_rows(), RNS.build_speed_rows()], ignore_index=True)
    tmp = BASE_ROW_CACHE.with_suffix(BASE_ROW_CACHE.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    rows.to_parquet(tmp, index=False)
    if BASE_ROW_CACHE.exists():
        BASE_ROW_CACHE.unlink()
    tmp.replace(BASE_ROW_CACHE)
    print(f"[cache] wrote {BASE_ROW_CACHE} rows={len(rows):,}", flush=True)


def ensure_enriched_row_cache() -> list[str]:
    ensure_base_row_cache()
    if ENRICHED_ROW_CACHE.exists():
        print(f"[cache] using {ENRICHED_ROW_CACHE}", flush=True)
        schema_cols = pq.ParquetFile(ENRICHED_ROW_CACHE).schema_arrow.names
        return sorted(c for c in schema_cols if c.startswith("ctx_"))

    rows = pd.read_parquet(BASE_ROW_CACHE)
    rows = add_merge_keys(rows, "origin_time")
    needed_times = set(pd.to_datetime(rows["origin_time"]).dt.normalize().unique())
    all_context_cols: list[str] = []
    enriched_parts: list[pd.DataFrame] = []
    for region in ("north_sea", "east_china_sea"):
        part = rows[rows["region"].eq(region)].copy()
        if part.empty:
            continue
        feat, context_cols = load_region_features(region, needed_times)
        all_context_cols = sorted(set(all_context_cols).union(context_cols))
        before = len(part)
        part = part.merge(feat, on=["_merge_time", "_lat_key", "_lon_key"], how="left", validate="many_to_one")
        matched = int(part[context_cols].notna().any(axis=1).sum()) if context_cols else 0
        print(f"[features] merged {region}: rows={before:,} matched={matched:,}", flush=True)
        enriched_parts.append(part)

    enriched = pd.concat(enriched_parts, ignore_index=True).sort_index(kind="mergesort")
    for col in all_context_cols:
        if col not in enriched.columns:
            enriched[col] = np.nan
    enriched = enriched.drop(columns=["_merge_time", "_lat_key", "_lon_key"], errors="ignore")
    tmp = ENRICHED_ROW_CACHE.with_suffix(ENRICHED_ROW_CACHE.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    enriched.to_parquet(tmp, index=False)
    if ENRICHED_ROW_CACHE.exists():
        ENRICHED_ROW_CACHE.unlink()
    tmp.replace(ENRICHED_ROW_CACHE)
    print(f"[cache] wrote {ENRICHED_ROW_CACHE} rows={len(enriched):,} ctx_cols={len(all_context_cols)}", flush=True)
    return all_context_cols


ORIGINAL_MAKE_FEATURES = LRN.make_features


def make_feature_rich_features(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if columns is None:
        base, base_cols = ORIGINAL_MAKE_FEATURES(df, None)
        ctx_cols = sorted(c for c in df.columns if c.startswith("ctx_"))
        ctx = df[ctx_cols].apply(pd.to_numeric, errors="coerce") if ctx_cols else pd.DataFrame(index=df.index)
        if ctx_cols:
            ctx = ctx.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        out = pd.concat([base, ctx], axis=1)
        cols = list(base_cols) + ctx_cols
        return out[cols], cols

    base, _ = ORIGINAL_MAKE_FEATURES(df, None)
    ctx_cols = sorted(c for c in df.columns if c.startswith("ctx_"))
    ctx = df[ctx_cols].apply(pd.to_numeric, errors="coerce") if ctx_cols else pd.DataFrame(index=df.index)
    if ctx_cols:
        ctx = ctx.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    out = pd.concat([base, ctx], axis=1)
    cols = list(columns)
    out = out.reindex(columns=cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out[cols], cols


def configure_learned_builder() -> None:
    LRN.ROW_CACHE = ENRICHED_ROW_CACHE
    LRN.OUT_CSV = OUT_CSV
    LRN.OUT_ZIP = OUT_ZIP
    LRN.CV_BY_FOLD_CSV = CV_BY_FOLD_CSV
    LRN.CV_SUMMARY_CSV = CV_SUMMARY_CSV
    LRN.DECISION_CSV = DECISION_CSV
    LRN.MANIFEST = MANIFEST
    LRN.model = lgbm_model
    LRN.make_features = make_feature_rich_features
    LRN.SEED = 20260614


def annotate_manifest(context_cols: list[str]) -> None:
    if not MANIFEST.exists():
        return
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["row_cache"] = str(ENRICHED_ROW_CACHE)
    data["base_row_cache"] = str(BASE_ROW_CACHE)
    data["sample_per_anchor_date"] = SAMPLE_PER_ANCHOR_DATE
    data["feature_rich_context_columns"] = context_cols
    data["gate_policy"]["direction_model"] = "LightGBM learned UV residual/direct-angle models over promoted-current-base rows plus full inference-available official feature vector"
    data["gate_policy"]["speed_model"] = "LightGBM quantile width model with q50 locked over promoted-current-base rows plus full inference-available official feature vector"
    data["wrapper"] = {
        "name": "feature_rich_newsignal_v1",
        "code_hashes": {
            "build_feature_rich_newsignal_v1_candidate.py": sha256(Path(__file__).resolve()),
            "build_learned_residual_newsignal_v1_candidate.py": sha256(ROOT / "build_learned_residual_newsignal_v1_candidate.py"),
            "build_regime_newsignal_v1_candidate.py": sha256(ROOT / "build_regime_newsignal_v1_candidate.py"),
        },
        "rule_note": (
            "Uses only columns present in both official train feature parquets and official inference feature parquets. "
            "Train-only target columns such as speed_d*/dir_d* are excluded by schema intersection."
        ),
    }
    data["competition_rule_notes"].append(
        "Feature-rich wrapper excludes train-only target columns by requiring every feature column to exist in official inference feature files."
    )
    data["code_hashes"]["wrapper"] = sha256(Path(__file__).resolve())
    MANIFEST.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    context_cols = ensure_enriched_row_cache()
    configure_learned_builder()
    LRN.main()
    annotate_manifest(context_cols)
    print(f"[manifest] {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
