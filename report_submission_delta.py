from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLS = [
    "type", "region", "station", "horizon", "hour", "level",
    "q05", "q50", "q95", "dir_05", "dir_50", "dir_95",
]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].fillna("").astype(str).str.lower().str.strip()
    df["region"] = df["region"].fillna("").astype(str)
    df["station"] = df["station"].fillna("").astype(str)
    df["level"] = df["level"].fillna("").astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    base_path = Path(args.base)
    cand_path = Path(args.candidate)
    print(f"Reading base: {base_path} ({base_path.stat().st_size:,} bytes)", flush=True)
    base = normalize(pd.read_csv(base_path, usecols=COLS, low_memory=False))
    print(f"Reading candidate: {cand_path} ({cand_path.stat().st_size:,} bytes)", flush=True)
    cand = normalize(pd.read_csv(cand_path, usecols=COLS, low_memory=False))
    if len(base) != len(cand):
        raise SystemExit(f"row count mismatch: base={len(base):,}, candidate={len(cand):,}")

    speed_changed = (base[SPEED_COLS].round(6) != cand[SPEED_COLS].round(6)).any(axis=1)
    dir_changed = (base[DIR_COLS].round(6) != cand[DIR_COLS].round(6)).any(axis=1)
    station = base["type"].eq("station")
    print(f"speed_changed_rows={int(speed_changed.sum()):,}", flush=True)
    print(f"dir_changed_rows={int(dir_changed.sum()):,}", flush=True)
    print(f"station_rows_changed={int((station & (speed_changed | dir_changed)).sum()):,}", flush=True)

    changed = cand.loc[speed_changed, ["region", "level", "horizon", "hour"] + SPEED_COLS].copy()
    for col in SPEED_COLS:
        changed[f"abs_delta_{col}"] = (cand.loc[speed_changed, col].astype(float) - base.loc[speed_changed, col].astype(float)).abs().values
    if changed.empty:
        print("No speed changes.", flush=True)
        return
    grouped = (
        changed.groupby(["region", "level", "horizon"], as_index=False)
        .agg(
            rows=("q50", "size"),
            q50_abs_mean=("abs_delta_q50", "mean"),
            q50_abs_p95=("abs_delta_q50", lambda x: x.quantile(0.95)),
            q05_abs_mean=("abs_delta_q05", "mean"),
            q95_abs_mean=("abs_delta_q95", "mean"),
        )
        .sort_values(["region", "level", "horizon"])
    )
    print(grouped.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
