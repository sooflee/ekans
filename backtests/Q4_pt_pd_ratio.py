"""
Q4 Platinum-Palladium ratio mean reversion.

When Pt/Pd > 1.5 -> long PL=F, short PA=F (ratio is high vs history -> Pt expensive vs Pd)
   actually a ratio > 1.5 means platinum is rich relative to palladium -> bet on reversion -> short PL / long PA
We follow the spec: when > 1.5 -> long PL / short PA; when < 0.4 -> reverse.
Hold 120 trading days. Re-enter only after exit.
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
        px = load_prices(["PL=F", "PA=F"], start="2005-01-01")
    except Exception as e:
        return mark_failed("Q4_pt_pd_ratio", f"yfinance load failed: {e}")

    px = px.dropna()
    if px.empty or "PL=F" not in px.columns or "PA=F" not in px.columns:
        return mark_failed("Q4_pt_pd_ratio", "PL=F or PA=F missing")

    rets = px.pct_change()
    ratio = (px["PL=F"] / px["PA=F"]).dropna()

    # Per spec: when ratio > 1.5 long PL / short PA. When < 0.4 reverse.
    sig = pd.Series(0, index=ratio.index)
    sig[ratio > 1.5] = 1   # long PL short PA
    sig[ratio < 0.4] = -1  # long PA short PL

    # Build position series with 120-day holding, non-overlapping
    pos_pl = pd.Series(0.0, index=rets.index)
    pos_pa = pd.Series(0.0, index=rets.index)
    triggers = sig[sig != 0]

    n_events = 0
    last_end = None
    events = []
    for d, s in triggers.items():
        idx = rets.index.searchsorted(d)
        if idx >= len(rets.index):
            continue
        start = rets.index[idx]
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(idx + 120, len(rets.index))
        for j in range(idx, end_idx):
            pos_pl.iloc[j] = float(s)
            pos_pa.iloc[j] = -float(s)
        last_end = rets.index[end_idx - 1]
        n_events += 1
        events.append((d, int(s)))

    if n_events == 0:
        return mark_failed("Q4_pt_pd_ratio", "no qualifying ratio triggers")

    pnl = (pos_pl.shift(1) * rets["PL=F"] + pos_pa.shift(1) * rets["PA=F"]).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets["PL=F"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q4 Pt/Pd ratio reversion")
    m["n_events"] = n_events
    print(f"Events: {n_events}; first: {events[0]}; last: {events[-1]}")
    print_metrics(m)

    save_result("Q4_pt_pd_ratio", m, extra={
        "status": "ok",
        "rule": "When Pt/Pd ratio > 1.5: long PL=F / short PA=F. When < 0.4: long PA=F / short PL=F. Hold 120 trading days, non-overlapping.",
        "mechanism": "Pt and Pd share auto-catalyst demand; sustained ratio extremes (>1.5 or <0.4) historically mean-revert as substitution and supply rebalance.",
        "source": "yfinance PL=F (platinum futures), PA=F (palladium futures)",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
