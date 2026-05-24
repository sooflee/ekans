"""
C14 TSMOM (Time-Series Momentum)
For a basket of SPY, TLT, GLD, DBC, EFA, EEM: monthly, if 12m return > 0, long that asset (1/N weight);
else flat that bucket. Aggregate PnL.
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
        return mark_failed("C14_tsmom", f"data load failed: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("C14_tsmom", "no overlap")

    monthly = px.resample("ME").last()
    r12 = monthly.pct_change(12)
    sig = (r12 > 0).astype(float) / len(tickers)

    sig_d = sig.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (sig_d * rets).sum(axis=1).dropna()

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C14 TSMOM (12m)")
    print_metrics(m)
    save_result("C14_tsmom", m, extra={
        "status": "ok",
        "rule": "Each asset: if 12m return > 0 hold long at 1/N, else flat that bucket. Monthly rebalance.",
        "universe": "SPY, TLT, GLD, DBC, EFA, EEM",
        "source": "Moskowitz-Ooi-Pedersen (2012) JFE 'Time Series Momentum'",
    })


if __name__ == "__main__":
    main()
