"""
J2 H.8 C&I loans contracting -> short SPY.

FRED BUSLOANS = Commercial & Industrial loans, monthly.
When two consecutive monthly readings both contract (negative MoM), short SPY for next month.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        df = load_fred("BUSLOANS", start="1990-01-01")
    except Exception as e:
        return mark_failed("J2_h8_ci_loans", f"FRED BUSLOANS load failed: {e}")
    if df.empty:
        return mark_failed("J2_h8_ci_loans", "Empty FRED load")

    s = df.iloc[:, 0].dropna()
    mom = s.pct_change()
    two_neg = (mom < 0) & (mom.shift(1) < 0)
    # signal arises at end of month t; we go short for month t+1
    trigger_dates = two_neg[two_neg].index

    px = load_prices(["SPY"], start="1995-01-01")
    if px.empty:
        return mark_failed("J2_h8_ci_loans", "SPY load failed")
    spy = px["SPY"].dropna()
    rets = spy.pct_change()

    daily_pos = pd.Series(1.0, index=rets.index)  # default long SPY (compare to long-SPY)
    n_events = 0
    for d in trigger_dates:
        # short for the calendar month following d
        # find first trading day on or after the first day of the month after d
        month_start = (d + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(1))
        month_end = month_start + pd.offsets.MonthEnd(1)
        mask = (rets.index >= month_start) & (rets.index <= month_end)
        if mask.any():
            daily_pos.loc[mask] = -1.0
            n_events += 1

    pnl = (daily_pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J2 C&I loans -> rotate SPY")
    print_metrics(m)
    print(f"\nTrigger months: {n_events}")

    save_result("J2_h8_ci_loans", m, extra={
        "status": "ok",
        "rule": "Two consecutive months of negative MoM in BUSLOANS -> short SPY for the next month; else long SPY.",
        "mechanism": "Falling bank C&I loans signal contracting business credit and weakening profits, headwind for equities.",
        "source": "https://fred.stlouisfed.org/series/BUSLOANS",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
