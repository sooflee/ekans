"""
AA1 XBI quarterly rebalance.
SSGA XBI is equal-weighted biotech; rebalances 3rd Friday Mar/Jun/Sep/Dec.
Hypothesis: heavy curation of constituent weights pre/post rebalance creates
a pricing distortion vs cap-weighted biotech (IBB).
Simple test: long XBI / short IBB in the 5-day window pre-rebalance (T-5 to T-1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calendar
import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def third_friday(year, month):
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2] if len(fridays) >= 3 else None


def main():
    try:
        px = load_prices(["XBI", "IBB"], start="2006-01-01")
    except Exception as e:
        return mark_failed("AA1_xbi_rebalance", f"data load failed: {e}")
    px = px.dropna()
    if len(px) < 200:
        return mark_failed("AA1_xbi_rebalance", "insufficient overlap XBI/IBB")

    rets = daily_returns(px)
    xbi_r = rets["XBI"]
    ibb_r = rets["IBB"]
    idx = rets.index

    # Build position: long XBI -1.0 short IBB on T-5..T-1 of each rebalance Fri.
    pos_xbi = pd.Series(0.0, index=idx)
    pos_ibb = pd.Series(0.0, index=idx)
    rebalance_months = [3, 6, 9, 12]
    years = range(idx[0].year, idx[-1].year + 1)
    n_events = 0
    for y in years:
        for m in rebalance_months:
            tf = third_friday(y, m)
            if tf is None:
                continue
            reb = pd.Timestamp(tf)
            i_reb = idx.searchsorted(reb, side="right") - 1
            if i_reb < 5:
                continue
            i_start = i_reb - 5
            i_end = i_reb - 1
            hold_dates = idx[i_start:i_end + 1]
            pos_xbi.loc[hold_dates] = 1.0
            pos_ibb.loc[hold_dates] = -1.0
            n_events += 1

    if n_events < 10:
        return mark_failed("AA1_xbi_rebalance",
                           f"too few rebalance events ({n_events})")

    # PnL: positions applied to next-day returns
    pnl = pos_xbi.shift(1) * xbi_r + pos_ibb.shift(1) * ibb_r
    pnl = pnl.dropna()
    # Benchmark = XBI buy-and-hold
    bench = xbi_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA1 XBI rebalance pair")
    print_metrics(m)
    save_result("AA1_xbi_rebalance", m, extra={
        "status": "ok",
        "rule": "Long XBI / short IBB from T-5 to T-1 of XBI rebalance (3rd Fri Mar/Jun/Sep/Dec).",
        "universe": "XBI vs IBB",
        "source": "SSGA XBI methodology; equal-weight vs cap-weight biotech",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
