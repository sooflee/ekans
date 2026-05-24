"""
Q2 Mexican Hacienda hedge window.

Hannover hedge (Mexico's annual oil hedge) typically executed Jun-Sep using long-dated
WTI put options. Rule: short CL=F for 6 weeks (30 trading days) when:
- date in Jun 1 - Sep 30
- OVX (CBOE crude vol) < 30
- WTI close > $70

Non-overlapping events; OVX from yfinance ^OVX.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        px = load_prices(["CL=F", "^OVX"], start="2007-05-01")
    except Exception as e:
        return mark_failed("Q2_mexican_hedge_window", f"yfinance load failed: {e}")
    px = px.dropna(how="all")
    if "CL=F" not in px.columns or "^OVX" not in px.columns:
        return mark_failed("Q2_mexican_hedge_window", "CL=F or ^OVX missing from yfinance")

    cl = px["CL=F"].dropna()
    ovx = px["^OVX"].dropna()
    rets = cl.pct_change()

    # Build trigger set: monthly first trigger to avoid daily clustering
    df = pd.concat([cl.rename("cl"), ovx.rename("ovx")], axis=1).dropna()
    cond = (df.index.month >= 6) & (df.index.month <= 9) & (df["ovx"] < 30) & (df["cl"] > 70)
    cand = df[cond]

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    events = []
    # iterate by year - only first qualifying day each year (avoid cluster)
    seen_years = set()
    for d, row in cand.iterrows():
        if d.year in seen_years:
            continue
        idx = rets.index.searchsorted(d)
        if idx >= len(rets.index):
            continue
        start = rets.index[idx]
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = -1.0  # short
        last_end = rets.index[end_idx - 1]
        n_events += 1
        events.append(d)
        seen_years.add(d.year)

    if n_events == 0:
        return mark_failed("Q2_mexican_hedge_window",
                           "no qualifying Jun-Sep windows with OVX<30 and CL>$70")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q2 Mexican hedge -> short CL=F 30d")
    m["n_events"] = n_events
    print(f"Events: {n_events}; dates: {[d.date() for d in events]}")
    print_metrics(m)

    save_result("Q2_mexican_hedge_window", m, extra={
        "status": "ok",
        "rule": "First trading day each year during Jun 1 - Sep 30 with OVX<30 and WTI close > $70: short CL=F for 30 trading days (~6 weeks).",
        "mechanism": "Mexican sovereign oil hedge (Hacienda) buys puts in low-vol summer; bank dealer delta-hedges short crude, pressuring spot for weeks.",
        "source": "yfinance CL=F (WTI front-month), ^OVX (CBOE crude oil VIX)",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
