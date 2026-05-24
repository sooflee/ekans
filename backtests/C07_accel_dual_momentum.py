"""
C07 Accelerating Dual Momentum
Avg of 1m, 3m, 6m total returns of SPY, SCZ (intl small), TLT.
Hold the single top-ranked asset monthly.
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
    tickers = ["SPY", "SCZ", "TLT"]
    try:
        px = load_prices(tickers, start="2007-12-01")
    except Exception as e:
        return mark_failed("C07_accel_dual_momentum", f"data load failed: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("C07_accel_dual_momentum", "no overlap")

    monthly = px.resample("ME").last()
    r1 = monthly.pct_change(1)
    r3 = monthly.pct_change(3)
    r6 = monthly.pct_change(6)
    accel = (r1 + r3 + r6) / 3.0

    pos = pd.DataFrame(0.0, index=monthly.index, columns=tickers)
    for d, row in accel.iterrows():
        if row.isna().any():
            continue
        winner = row.idxmax()
        pos.loc[d, winner] = 1.0

    pos_d = pos.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (pos_d * rets).sum(axis=1).dropna()

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C07 Accelerating Dual Momentum")
    print_metrics(m)
    save_result("C07_accel_dual_momentum", m, extra={
        "status": "ok",
        "rule": "Monthly: avg(1m,3m,6m) return across {SPY,SCZ,TLT}; hold top asset.",
        "universe": "SPY, SCZ, TLT",
        "source": "Engineered Portfolio / Accelerating Dual Momentum (2018)",
    })


if __name__ == "__main__":
    main()
