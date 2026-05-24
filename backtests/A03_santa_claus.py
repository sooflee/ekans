"""
A03 Santa Claus Rally
Long SPY from close of 5th-to-last trading day of Dec through close of 2nd trading day of Jan.
Cash otherwise.
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
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    df = pd.DataFrame({"ret": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month

    pos = pd.Series(0.0, index=idx)

    # For each calendar year, build holding window:
    # Position-hold dates = last 5 trading days of Dec (Dec T-4 .. Dec T-0) and first trading day of Jan
    # so that returns are earned for Dec T-3..T-0 and Jan trading days 1 and 2.
    # Rule says "from close of 5th-to-last day of Dec through close of 2nd trading day of Jan".
    # That = returns earned on the 4 days following the 5th-to-last (i.e., last 4 Dec days)
    # plus first 2 Jan trading days. To hold returns at last-Dec-day t, we need pos at t-1.
    # Easy: union of (last 5 trading days of Dec) and (first trading day of Jan).
    years = sorted(df["year"].unique())
    for y in years:
        dec = df[(df["year"] == y) & (df["month"] == 12)]
        if len(dec) >= 5:
            for d in dec.index[-5:]:
                pos.loc[d] = 1.0
        jan = df[(df["year"] == y + 1) & (df["month"] == 1)] if (y + 1) in years else df[(df["year"] == y) & (df["month"] == 1)]
        # Use jan of next year if available
        jan_next = df[(df["year"] == y + 1) & (df["month"] == 1)]
        if len(jan_next) >= 1:
            pos.loc[jan_next.index[0]] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A03 Santa Claus")
    print_metrics(m)
    save_result("A03_santa_claus", m, extra={
        "status": "ok",
        "rule": "Long SPY from close of 5th-to-last trading day of Dec through close of 2nd trading day of Jan.",
        "universe": "SPY",
        "source": "Hirsch (Stock Trader's Almanac)",
    })


if __name__ == "__main__":
    main()
