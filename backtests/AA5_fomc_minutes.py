"""
AA5 FOMC minutes drift.
FOMC minutes released ~3 weeks after the FOMC meeting (8/year).
Hypothesis: minutes contain hawkish/dovish detail that moves the front end of
the curve. Spec: short 2Y futures via TBT proxy on minutes day, daily close PnL.

TBT = ProShares Ultra Short 20+Y Treasury (long TBT = short long bonds).
Using TBT as a proxy for short-duration exposure on minutes day.
Hold from prior close to minutes-day close (capture the release in a single day).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)
from _fomc_dates import FOMC_DATES


def add_business_days(d, n):
    return pd.bdate_range(d, periods=n + 1)[-1]


def main():
    try:
        px = load_prices(["TBT"], start="2008-01-01")
    except Exception as e:
        return mark_failed("AA5_fomc_minutes", f"data load failed: {e}")
    tbt = px["TBT"].dropna()
    if len(tbt) < 200:
        return mark_failed("AA5_fomc_minutes", "insufficient TBT data")
    idx = tbt.index
    tbt_r = tbt.pct_change()

    # Minutes are released exactly 3 weeks after the FOMC date.
    minute_dates = []
    for d_str in FOMC_DATES:
        d = pd.Timestamp(d_str)
        m_date = d + pd.Timedelta(days=21)
        # Move to the next business day (Mon-Fri) if it falls on weekend.
        if m_date.weekday() >= 5:
            m_date = m_date + pd.Timedelta(days=(7 - m_date.weekday()))
        minute_dates.append(m_date)

    pos = pd.Series(0.0, index=idx)
    n_events = 0
    for m_date in minute_dates:
        if m_date < idx[0] or m_date > idx[-1]:
            continue
        i = idx.searchsorted(m_date, side="right") - 1
        if i < 1:
            continue
        # Long TBT *on* the minutes day. Position set at prior close (i-1).
        pos.iloc[i - 1] = 1.0
        n_events += 1

    if n_events < 20:
        return mark_failed("AA5_fomc_minutes", f"too few events ({n_events})")

    pnl = (pos.shift(1) * tbt_r).dropna()
    bench = tbt_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA5 FOMC minutes -> TBT day")
    print_metrics(m)
    save_result("AA5_fomc_minutes", m, extra={
        "status": "ok",
        "rule": "Long TBT (short long-duration UST) on FOMC minutes release day (FOMC + 21 calendar days).",
        "universe": "TBT (UST proxy)",
        "source": "Federal Reserve FOMC minutes calendar",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
