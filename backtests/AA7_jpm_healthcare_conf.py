"""
AA7 JPM Healthcare Conference drift.
JPM Healthcare Conference = 2nd Mon of January (largest healthcare conference;
biotech CEO presentations, deal announcements). Historically biotech rallies into
and through the conference.

Spec: long XBI from 2nd Mon of December → 2nd Fri of January (~5 weeks).
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


def nth_weekday(year, month, weekday, n):
    """nth occurrence (1-indexed) of weekday in (year, month). weekday: Mon=0..Sun=6."""
    cal = calendar.Calendar(firstweekday=0)
    days = [d for d in cal.itermonthdates(year, month)
            if d.month == month and d.weekday() == weekday]
    return days[n - 1] if len(days) >= n else None


def main():
    try:
        px = load_prices(["XBI"], start="2006-01-01")
    except Exception as e:
        return mark_failed("AA7_jpm_healthcare_conf", f"data load failed: {e}")
    xbi = px["XBI"].dropna()
    rets = xbi.pct_change()
    idx = xbi.index

    pos = pd.Series(0.0, index=idx)
    n_events = 0
    for y in range(idx[0].year, idx[-1].year + 1):
        # Entry: 2nd Monday of December year y-1 (event year y conference is in Jan y)
        # Spec: "long XBI from 2nd Mon December through 2nd Fri January (≈5 weeks)"
        # Entry year = December of year (y-1); exit year = January of year y.
        dec_mon = nth_weekday(y - 1, 12, 0, 2)
        jan_fri = nth_weekday(y, 1, 4, 2)
        if dec_mon is None or jan_fri is None:
            continue
        d_in = pd.Timestamp(dec_mon)
        d_out = pd.Timestamp(jan_fri)
        if d_out < idx[0] or d_in > idx[-1]:
            continue
        i_in = idx.searchsorted(d_in, side="left")  # first trading day >= entry
        i_out = idx.searchsorted(d_out, side="right") - 1  # last trading day <= exit
        if i_in < 1 or i_out >= len(idx) or i_in >= i_out:
            continue
        # Position from i_in to i_out inclusive (set at close of i_in-1, earned through i_out)
        hold = idx[i_in - 1:i_out]  # set position at i_in-1..i_out-1, earns next-day return
        pos.loc[hold] = 1.0
        n_events += 1

    if n_events < 5:
        return mark_failed("AA7_jpm_healthcare_conf", f"too few events ({n_events})")

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA7 JPM Healthcare drift -> XBI")
    print_metrics(m)
    save_result("AA7_jpm_healthcare_conf", m, extra={
        "status": "ok",
        "rule": "Long XBI from 2nd Mon Dec → 2nd Fri Jan (JPM Healthcare Conference drift).",
        "universe": "XBI",
        "source": "JPM Healthcare Conference (~2nd Mon Jan), biotech rally narrative",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
