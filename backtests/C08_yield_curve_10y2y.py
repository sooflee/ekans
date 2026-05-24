"""
C08 Yield curve inversion (10Y-2Y)
FRED T10Y2Y. When 10Y-2Y crosses below 0 from above, set 12-month timer; on month 12
post-inversion, reduce equity to 50% (or cash) until curve re-steepens above +50bps.
Compare to buy-and-hold.
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
        t10y2y = load_fred("T10Y2Y", start="1990-01-01").iloc[:, 0].rename("T10Y2Y")
        spy = load_prices(["SPY"], start="1993-01-29").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("C08_yield_curve_10y2y", f"data load failed: {e}")

    t10y2y = t10y2y.reindex(spy.index, method="ffill")
    df = pd.concat([spy, t10y2y], axis=1).dropna()

    # Detect inversion crosses below 0
    spread = df["T10Y2Y"]
    prev = spread.shift(1)
    inv_cross = (prev > 0) & (spread <= 0)

    inv_dates = df.index[inv_cross]

    pos = pd.Series(1.0, index=df.index)
    # For each inversion event, start the 12-month timer
    for d in inv_dates:
        start_idx = df.index.searchsorted(d)
        # exactly 12 months later (~252 trading days)
        target = d + pd.DateOffset(months=12)
        red_idx = df.index.searchsorted(target)
        if red_idx >= len(df.index):
            continue
        # From that date forward, reduce equity to 50% until spread > +0.5
        i = red_idx
        while i < len(df.index):
            if spread.iloc[i] > 0.5:
                break
            pos.iloc[i] = min(pos.iloc[i], 0.5)
            i += 1

    rets = df["SPY"].pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C08 Yield curve (10Y-2Y) inversion timer")
    print_metrics(m)
    save_result("C08_yield_curve_10y2y", m, extra={
        "status": "ok",
        "rule": "On 10Y-2Y crossing below 0, set 12-month timer; from month 12, reduce SPY to 50% until spread >+0.5.",
        "universe": "SPY (gated by T10Y2Y)",
        "source": "Estrella & Mishkin; Campbell Harvey thesis",
    })


if __name__ == "__main__":
    main()
