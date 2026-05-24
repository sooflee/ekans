"""
C15 Trend-following (Hurst-Ooi-Pedersen)
Same basket as C14 but use blend of 1m + 3m + 12m return signs (each weighted 1/3).
Allow short positions when signal negative.
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


def main():
    tickers = ["SPY", "TLT", "GLD", "DBC", "EFA", "EEM"]
    try:
        px = load_prices(tickers, start="2003-01-01")
    except Exception as e:
        return mark_failed("C15_trend_following", f"data load failed: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("C15_trend_following", "no overlap")

    monthly = px.resample("ME").last()
    r1 = monthly.pct_change(1)
    r3 = monthly.pct_change(3)
    r12 = monthly.pct_change(12)

    s1 = np.sign(r1)
    s3 = np.sign(r3)
    s12 = np.sign(r12)
    blended = (s1 + s3 + s12) / 3.0
    sig = blended / len(tickers)  # weight per asset

    sig_d = sig.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (sig_d * rets).sum(axis=1).dropna()

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C15 Trend-following (1m+3m+12m signs)")
    print_metrics(m)
    save_result("C15_trend_following", m, extra={
        "status": "ok",
        "rule": "Per asset: avg sign of 1m,3m,12m returns -> long/short at that fractional weight (1/N).",
        "universe": "SPY, TLT, GLD, DBC, EFA, EEM (long/short allowed)",
        "source": "Hurst-Ooi-Pedersen (2017) JPM 'A Century of Evidence on Trend-Following'",
    })


if __name__ == "__main__":
    main()
