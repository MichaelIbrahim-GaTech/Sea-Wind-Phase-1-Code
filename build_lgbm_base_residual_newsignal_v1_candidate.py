from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor

import build_learned_residual_newsignal_v1_candidate as LRN
import build_regime_newsignal_v1_candidate as RNS


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "v6_pressure_speed"

SAMPLE_PER_ANCHOR_DATE = 180
ROW_CACHE = WORK / f"regime_newsignal_v1_rows_s{SAMPLE_PER_ANCHOR_DATE}.parquet"
OUT_CSV = WORK / "pred_lgbm_base_residual_newsignal_v1.csv"
OUT_ZIP = WORK / "sub_lgbmbres_v1.zip"
CV_BY_FOLD_CSV = WORK / "cv_lgbm_base_residual_newsignal_v1_by_fold.csv"
CV_SUMMARY_CSV = WORK / "cv_lgbm_base_residual_newsignal_v1_summary.csv"
DECISION_CSV = WORK / "decision_lgbm_base_residual_newsignal_v1.csv"
MANIFEST = WORK / "manifest_lgbm_base_residual_newsignal_v1.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=7,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.08,
        reg_lambda=4.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def ensure_row_cache() -> None:
    if ROW_CACHE.exists():
        print(f"[cache] using {ROW_CACHE}", flush=True)
        return
    print(f"[cache] building {ROW_CACHE}", flush=True)
    RNS.SAMPLE_PER_ANCHOR_DATE = SAMPLE_PER_ANCHOR_DATE
    RNS.ROW_CACHE = ROW_CACHE
    RNS.install_fast_anchor_predictors()
    rows = pd.concat([RNS.build_direction_rows(), RNS.build_speed_rows()], ignore_index=True)
    tmp = ROW_CACHE.with_suffix(ROW_CACHE.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    rows.to_parquet(tmp, index=False)
    if ROW_CACHE.exists():
        ROW_CACHE.unlink()
    tmp.replace(ROW_CACHE)
    print(f"[cache] wrote {ROW_CACHE} rows={len(rows):,}", flush=True)


def configure_learned_builder() -> None:
    LRN.ROW_CACHE = ROW_CACHE
    LRN.OUT_CSV = OUT_CSV
    LRN.OUT_ZIP = OUT_ZIP
    LRN.CV_BY_FOLD_CSV = CV_BY_FOLD_CSV
    LRN.CV_SUMMARY_CSV = CV_SUMMARY_CSV
    LRN.DECISION_CSV = DECISION_CSV
    LRN.MANIFEST = MANIFEST
    LRN.model = lgbm_model
    LRN.SEED = 20260614


def annotate_manifest() -> None:
    if not MANIFEST.exists():
        return
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["row_cache"] = str(ROW_CACHE)
    data["sample_per_anchor_date"] = SAMPLE_PER_ANCHOR_DATE
    data["gate_policy"]["direction_model"] = "LightGBM learned UV residual/direct-angle models over promoted-current-base official anchor rows"
    data["gate_policy"]["speed_model"] = "LightGBM quantile width model with q50 locked over promoted-current-base official anchor rows"
    data["wrapper"] = {
        "name": "lgbm_base_residual_newsignal_v1",
        "code_hashes": {
            "build_lgbm_base_residual_newsignal_v1_candidate.py": sha256(Path(__file__).resolve()),
            "build_learned_residual_newsignal_v1_candidate.py": sha256(ROOT / "build_learned_residual_newsignal_v1_candidate.py"),
            "build_regime_newsignal_v1_candidate.py": sha256(ROOT / "build_regime_newsignal_v1_candidate.py"),
        },
        "rule_note": "Uses the same current-base anchor rows and strict gates as learned_residual_newsignal_v1, with LightGBM replacing HistGradientBoosting.",
    }
    data["code_hashes"]["wrapper"] = sha256(Path(__file__).resolve())
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ensure_row_cache()
    configure_learned_builder()
    LRN.main()
    annotate_manifest()
    print(f"[manifest] {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
