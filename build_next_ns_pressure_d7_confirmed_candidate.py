from __future__ import annotations

from pathlib import Path

import pandas as pd

import sea_winds_end_to_end_final as E2E


WORK = Path("runs/v6_pressure_speed")
BASE_CSV = WORK / "predictions_public_positive_fullrefit_hybrid_compact.csv"
OUT_CSV = WORK / "predictions_next_ns_pressure_d7_confirmed_compact.csv"
OUT_ZIP = WORK / "submission_next_ns_pressure_d7_confirmed_compact.zip"


def main() -> None:
    print(f"Reading locked base {BASE_CSV} ({BASE_CSV.stat().st_size:,} bytes)", flush=True)
    df = pd.read_csv(BASE_CSV, low_memory=False)[E2E.COLS]
    df = E2E.normalize_for_assembly(df)

    before = df[E2E.COLS].copy()
    changed_rows = E2E.apply_ns_pressure_d7_hres(df)
    final = E2E.validate_final(df)

    after = final[E2E.COLS]
    speed_changed = (before[E2E.SPEED_COLS].round(2).to_numpy() != after[E2E.SPEED_COLS].round(2).to_numpy()).any(axis=1).sum()
    dir_changed = (before[E2E.DIR_COLS].round(1).to_numpy() != after[E2E.DIR_COLS].round(1).to_numpy()).any(axis=1).sum()
    print(f"Applied NS pressure d7 HRES direction rows: {changed_rows:,}", flush=True)
    print(f"Delta vs locked base: speed_rows={int(speed_changed):,}; dir_rows={int(dir_changed):,}", flush=True)
    if int(speed_changed) != 0 or int(dir_changed) != int(changed_rows):
        raise SystemExit("unexpected delta outside the target direction block")

    E2E.write_zip(final, OUT_CSV, OUT_ZIP)
    print(f"OK: {OUT_ZIP}", flush=True)


if __name__ == "__main__":
    main()
