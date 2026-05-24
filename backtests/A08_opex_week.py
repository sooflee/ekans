"""
A08 Monthly OPEX week.
Long SPY from close of Friday before OPEX week (i.e., the Friday before the 3rd Friday)
through close of OPEX Friday (3rd Friday of the month).
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


def third_friday(year, month):
    """Return the date of the 3rd Friday of (year, month)."""
    import calendar
    # weekday(): Monday=0..Friday=4
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2] if len(fridays) >= 3 else None


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    # For each (year, month), find third Friday and the Friday before.
    pos = pd.Series(0.0, index=idx)
    years = range(idx[0].year, idx[-1].year + 1)
    for y in years:
        for m in range(1, 13):
            tf = third_friday(y, m)
            if tf is None:
                continue
            opex = pd.Timestamp(tf)
            prior_friday = opex - pd.Timedelta(days=7)
            # Map to nearest trading day at or before
            i_opex = idx.searchsorted(opex, side="right") - 1
            i_prior = idx.searchsorted(prior_friday, side="right") - 1
            if i_opex < 0 or i_prior < 0 or i_prior >= i_opex:
                continue
            # Hold from close of prior_friday through close of opex.
            # Pos at i_prior..(i_opex - 1) earns returns on (i_prior+1)..i_opex.
            hold_dates = idx[i_prior:i_opex]
            pos.loc[hold_dates] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A08 OPEX week")
    print_metrics(m)
    save_result("A08_opex_week", m, extra={
        "status": "ok",
        "rule": "Long SPY from close of Friday before OPEX week through close of OPEX Friday (3rd Friday).",
        "universe": "SPY",
        "source": "Stivers-Sun, options-OPEX literature",
    })


if __name__ == "__main__":
    main()
