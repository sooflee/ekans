"""
Backfill IS/OOS metrics on legacy result files from their cached PnL parquets.

The harness started recording `is_sharpe`/`oos_sharpe`/`is_cagr`/`oos_cagr` only
recently. Older result files (~88% of the catalog) lack these fields, which
means `pipeline/winner_gate.is_winner` rejects them on the `oos_sharpe > 0`
criterion regardless of their actual performance.

For backtests that saved their PnL series to `results/pnl/<sid>.parquet`, we
can recompute IS/OOS metrics offline without re-running the original backtest
script — much cheaper than a full catalog re-run.

Result files are updated atomically (write tmp + os.replace). Existing fields
are preserved; only the IS/OOS group is added.

Usage:
    .venv/bin/python pipeline/backfill_oos.py            # dry-run summary
    .venv/bin/python pipeline/backfill_oos.py --apply    # write into results/*.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PNL_DIR = RESULTS_DIR / "pnl"

ANN_FACTOR = 252
MIN_HALF_LEN = 60  # matches harness.compute_metrics


def is_oos_metrics(pnl: pd.Series) -> dict | None:
    """Compute the IS/OOS chunk metrics on the given daily PnL series.

    Matches the logic in backtests/harness.py:compute_metrics so backfilled
    numbers are identical to what a fresh run would have produced.
    Returns None if the series is too short to split.
    """
    pnl = pnl.dropna()
    mid = len(pnl) // 2
    if mid < MIN_HALF_LEN:
        return None
    out: dict = {}
    for label, chunk in [("is", pnl.iloc[:mid]), ("oos", pnl.iloc[mid:])]:
        if len(chunk) == 0:
            continue
        c_eq = (1 + chunk).cumprod()
        c_years = len(chunk) / ANN_FACTOR
        c_cagr = c_eq.iloc[-1] ** (1 / c_years) - 1
        if chunk.std() > 0:
            c_sharpe = chunk.mean() / chunk.std() * np.sqrt(ANN_FACTOR)
        else:
            c_sharpe = 0.0
        out[f"{label}_cagr"] = float(c_cagr)
        out[f"{label}_sharpe"] = float(c_sharpe)
        out[f"{label}_start"] = str(chunk.index[0].date())
        out[f"{label}_end"] = str(chunk.index[-1].date())
    return out


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backfill_one(sid: str, apply_changes: bool) -> tuple[str, dict | None]:
    """Return (action, metrics) where action is one of:
      skip_no_result | skip_already_has_oos | skip_no_parquet |
      skip_short_pnl | skip_bad_pnl | filled
    """
    result_path = RESULTS_DIR / f"{sid}.json"
    pnl_path = PNL_DIR / f"{sid}.parquet"
    if not result_path.exists():
        return "skip_no_result", None
    if not pnl_path.exists():
        return "skip_no_parquet", None
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        return "skip_no_result", None
    if result.get("oos_sharpe") is not None:
        return "skip_already_has_oos", None
    try:
        df = pd.read_parquet(pnl_path)
    except Exception:
        return "skip_bad_pnl", None
    if "pnl" in df.columns:
        pnl = df["pnl"]
    elif df.shape[1] == 1:
        pnl = df.iloc[:, 0]
    else:
        return "skip_bad_pnl", None
    pnl = pd.to_numeric(pnl, errors="coerce")
    metrics = is_oos_metrics(pnl)
    if metrics is None:
        return "skip_short_pnl", None
    if apply_changes:
        result.update(metrics)
        _atomic_write_json(result_path, result)
    return "filled", metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write IS/OOS fields into results/*.json")
    args = ap.parse_args()

    sids = sorted(p.stem for p in PNL_DIR.glob("*.parquet"))
    print(f"Found {len(sids)} cached PnL parquets in {PNL_DIR}")

    counts: dict[str, int] = {}
    filled_sample: list[tuple[str, dict]] = []
    for sid in sids:
        action, metrics = backfill_one(sid, args.apply)
        counts[action] = counts.get(action, 0) + 1
        if action == "filled" and len(filled_sample) < 8:
            filled_sample.append((sid, metrics or {}))

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"\n=== Backfill summary [{mode}] ===")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    if filled_sample:
        print("\nSample of filled entries:")
        for sid, m in filled_sample:
            print(f"  {sid}: is_sharpe={m.get('is_sharpe'):.2f}  oos_sharpe={m.get('oos_sharpe'):.2f}  "
                  f"is_cagr={(m.get('is_cagr') or 0)*100:.1f}%  oos_cagr={(m.get('oos_cagr') or 0)*100:.1f}%")

    if not args.apply:
        print("\n(dry run; pass --apply to write into results/*.json)")
    else:
        print("\nNext step: rerun the BH table so newly-OOS-eligible signals propagate:")
        print("  .venv/bin/python pipeline/refresh_bh.py")


if __name__ == "__main__":
    main()
