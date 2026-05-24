"""
P11 Karsan Year-End Recollateralization.

Conditional Santa Rally:
- Compute SPY YTD return from Jan 2 close through Oct 31 close.
- If YTD > +5%, long SPY from Nov 1 close (next trading day) to Jan 31 close.
- Else flat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["SPY"], start="2000-01-01")["SPY"]
    rets = px.pct_change()
    idx = px.index

    pos = pd.Series(0.0, index=idx)
    years = sorted(set(idx.year))
    n_trig = 0
    n_total = 0
    for y in years:
        # Find first trading day of year and last trading day on/before Oct 31.
        year_mask = idx.year == y
        if not year_mask.any():
            continue
        ytd_start = idx[year_mask][0]
        oct_end_target = pd.Timestamp(f"{y}-10-31")
        ix_oct = idx.searchsorted(oct_end_target, side="right") - 1
        if ix_oct < 0 or idx[ix_oct].year != y:
            continue
        ytd_ret = px.iloc[ix_oct] / px.loc[ytd_start] - 1
        n_total += 1
        if ytd_ret <= 0.05:
            continue
        n_trig += 1
        # Hold from Nov 1 close through Jan 31 close of year+1.
        nov1 = pd.Timestamp(f"{y}-11-01")
        ix_nov1 = idx.searchsorted(nov1, side="right") - 1
        if ix_nov1 < 0:
            continue
        jan31 = pd.Timestamp(f"{y+1}-01-31")
        ix_jan31 = idx.searchsorted(jan31, side="right") - 1
        if ix_jan31 <= ix_nov1:
            continue
        # Position at i_nov1..(i_jan31-1) earns returns on (i_nov1+1)..i_jan31
        hold_dates = idx[ix_nov1:ix_jan31]
        pos.loc[hold_dates] = 1.0

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets.dropna(), name="P11 Karsan Year-End Recollat.")
    print_metrics(m)
    save_result("P11_karsan_year_end_recollateralization", m, extra={
        "status": "ok",
        "rule": "If SPY YTD through Oct 31 > +5%, long SPY from Nov 1 close to Jan 31 close; else flat.",
        "mechanism": "Karsan: levered-fund/dealer recollateralization + window-dressing into year-end when YTD is up.",
        "universe": "SPY",
        "n_triggered_years": n_trig,
        "n_total_years": n_total,
        "source": "Cem Karsan (YouTube/Twitter, Phase 1P)",
    })


if __name__ == "__main__":
    main()
