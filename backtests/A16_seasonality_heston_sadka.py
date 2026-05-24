"""
A16 Seasonality (Heston-Sadka same-month, single-asset version on SPY).
For each month, compute that month's historical avg daily return over the prior 10
calendar years. If avg > 0, hold SPY for that month; else cash. Walk-forward (no
look-ahead): the decision for month M of year Y uses only data through year Y-1.
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
    # Start earlier to get a 10y burn-in
    px = load_prices(["SPY"], start="1993-02-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    df = pd.DataFrame({"ret": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month

    pos = pd.Series(0.0, index=idx)
    LOOKBACK = 10  # years

    start_year = df["year"].min() + LOOKBACK
    end_year = df["year"].max()

    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            # Look at months m in years [y-LOOKBACK, y-1]
            hist = df[(df["year"] >= y - LOOKBACK) & (df["year"] <= y - 1) & (df["month"] == m)]
            if len(hist) < 30:
                continue
            avg = hist["ret"].mean()
            if avg > 0:
                month_dates = df[(df["year"] == y) & (df["month"] == m)].index
                pos.loc[month_dates] = 1.0

    # Only evaluate over the live period (post burn-in)
    live_idx = idx[idx.year >= start_year]
    pnl = long_short_pnl(pos, rets).loc[live_idx]
    bench = rets.loc[live_idx]

    m = compute_metrics(pnl, benchmark=bench, name="A16 Heston-Sadka SPY")
    print_metrics(m)
    save_result("A16_seasonality_heston_sadka", m, extra={
        "status": "ok",
        "rule": "For each month, if SPY's prior 10y same-month avg daily return > 0, hold SPY that month; else cash. Walk-forward.",
        "universe": "SPY",
        "source": "Heston & Sadka (JFE 2008) — single-asset reduced-form version",
        "lookback_years": LOOKBACK,
        "live_start": str(live_idx[0].date()) if len(live_idx) > 0 else None,
    })


if __name__ == "__main__":
    main()
