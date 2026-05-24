"""
O10 Census Business Formation Statistics weekly business applications.

DATA SUBSTITUTION: FRED hosts the official Census BFS weekly Business Applications
NSA series as BUSAPPWNSAUS (2006-present). The 'BA_BAHPC' high-propensity sub-series
is not exposed on FRED; using all-applications (BA_BA) instead - this is the
broadest, most reliable signal anyway.

When YoY % change (52w) > +5% for 4 consecutive weeks, long IWM for 90 trading days.
Mechanism: New-business formation surges signal entrepreneurial risk-on, often
preceding small-cap leadership; small-business hiring & spending feeds Russell-2000
revenues.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        ba = load_fred("BUSAPPWNSAUS", start="2005-01-01").iloc[:, 0].rename("BA")
        iwm = load_prices(["IWM"], start="2005-01-01").iloc[:, 0].rename("IWM")
        spy = load_prices(["SPY"], start="2005-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O10_business_formation", f"data load failed: {e}")

    yoy = ba.pct_change(52)
    print(f"YoY stats: median={yoy.median()*100:.2f}%, max={yoy.max()*100:.2f}%, min={yoy.min()*100:.2f}%")

    # 4 consecutive weeks with YoY > 5%
    mask = yoy > 0.05
    # streak: True only when last 4 weeks all hit
    streak4 = mask & mask.shift(1, fill_value=False) & mask.shift(2, fill_value=False) & mask.shift(3, fill_value=False)
    # First trigger of each spell
    first = streak4 & ~streak4.shift(1, fill_value=False)
    triggers = yoy.index[first.fillna(False)]
    n_events = len(triggers)

    iwm_rets = iwm.pct_change()
    pos = pd.Series(0.0, index=iwm_rets.index)
    hold = 90
    for d in triggers:
        loc = iwm_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 3:
        return mark_failed("O10_business_formation", f"only {n_events} events", extra={"n_events": int(n_events)})

    pnl = (pos * iwm_rets).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O10 BFS YoY>5% (4w) -> long IWM 90d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O10_business_formation", m, extra={
        "status": "ok",
        "rule": "When weekly business applications YoY (52w) > +5% for 4 consecutive weeks, long IWM 90 sessions.",
        "mechanism": "New-business formation surges precede small-cap leadership / small-biz revenue.",
        "universe": "IWM",
        "source": "FRED BUSAPPWNSAUS (Census BFS weekly, all business applications, NSA).",
        "n_events": int(n_events),
        "data_substitution": "Used all-applications (BA_BA) since BFS BA_BAHPC high-propensity sub-series is not on FRED.",
    })


if __name__ == "__main__":
    main()
