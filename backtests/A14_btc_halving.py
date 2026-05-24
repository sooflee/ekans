"""
A14 BTC halving cycle.
Long BTC-USD from 6 months before each halving through 18 months after; flat otherwise.
Halvings: Nov 28 2012, Jul 9 2016, May 11 2020, Apr 19 2024.
BTC-USD on yfinance starts ~2014, so the 2012 halving will be missed.
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


HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
]


def main():
    px = load_prices(["BTC-USD"], start="2014-01-01")
    rets = daily_returns(px)
    # BTC-USD column name handling
    col = rets.columns[0]
    r = rets[col].dropna()
    idx = r.index

    pos = pd.Series(0.0, index=idx)
    for h in HALVINGS:
        lo = h - pd.DateOffset(months=6)
        hi = h + pd.DateOffset(months=18)
        mask = (idx >= lo) & (idx <= hi)
        pos.loc[mask] = 1.0

    pnl = (pos.shift(1) * r).dropna()
    m = compute_metrics(pnl, benchmark=r, name="A14 BTC halving cycle")
    print_metrics(m)
    save_result("A14_btc_halving", m, extra={
        "status": "ok",
        "rule": "Long BTC-USD from H-6mo through H+18mo around each halving; flat otherwise.",
        "universe": "BTC-USD",
        "source": "Bitcoin halving narrative (no academic citation; Plan B / NVT discourse)",
        "halvings_used": [str(h.date()) for h in HALVINGS],
        "note": "2012 halving missed: BTC-USD on yfinance starts ~Sep 2014.",
    })


if __name__ == "__main__":
    main()
