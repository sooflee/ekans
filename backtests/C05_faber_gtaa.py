"""
C05 Faber GTAA 10-mo MA
Five assets: SPY, EFA, VNQ, GSG (commodities), IEF (10Y Treasuries).
Monthly close: if close > 10-month SMA, hold; else cash for that bucket. Equal-weight active buckets.
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
    tickers = ["SPY", "EFA", "VNQ", "GSG", "IEF"]
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed("C05_faber_gtaa", f"data load failed: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("C05_faber_gtaa", "no overlap among GTAA universe")

    # Monthly close
    monthly = px.resample("ME").last()
    sma10 = monthly.rolling(10).mean()
    signal_m = (monthly > sma10).astype(float)  # 1 if hold, 0 if cash

    # Equal weight active buckets: weight = (1/N) * signal
    N = len(tickers)
    weights_m = signal_m / N

    # Reindex to daily, forward-fill, and shift 1 day to apply at next bar
    weights_d = weights_m.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()

    pnl = (weights_d * rets).sum(axis=1).dropna()
    bench_rets = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench_rets, name="C05 Faber GTAA (10mo MA)")
    print_metrics(m)
    save_result("C05_faber_gtaa", m, extra={
        "status": "ok",
        "rule": "Monthly: each asset on if close > 10mo SMA; equal-weight active buckets (1/5 each).",
        "universe": "SPY, EFA, VNQ, GSG, IEF",
        "source": "Faber (2007) 'A Quantitative Approach to Tactical Asset Allocation'",
    })


if __name__ == "__main__":
    main()
