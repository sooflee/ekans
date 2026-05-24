"""
C06 Dual Momentum (Antonacci GEM)
Monthly: 12-month total return of SPY, ACWX (intl ex-US), AGG (bonds).
If max(SPY 12m, ACWX 12m) > BIL 12m return, hold the winner; else hold AGG.
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
    tickers = ["SPY", "ACWX", "AGG", "BIL"]
    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed("C06_dual_momentum", f"data load failed: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("C06_dual_momentum", "no overlap")

    monthly = px.resample("ME").last()
    r12 = monthly.pct_change(12)

    # Decision frame
    pos = pd.DataFrame(0.0, index=monthly.index, columns=tickers)
    for d, row in r12.iterrows():
        if row.isna().any():
            continue
        # Choose between SPY and ACWX (the equity champion)
        equity_winner = "SPY" if row["SPY"] >= row["ACWX"] else "ACWX"
        if row[equity_winner] > row["BIL"]:
            pos.loc[d, equity_winner] = 1.0
        else:
            pos.loc[d, "AGG"] = 1.0

    # daily PnL
    pos_d = pos.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (pos_d * rets).sum(axis=1).dropna()

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C06 Dual Momentum (GEM)")
    print_metrics(m)
    save_result("C06_dual_momentum", m, extra={
        "status": "ok",
        "rule": "Monthly: 12m total return of SPY,ACWX. Pick equity winner if > BIL 12m, else hold AGG.",
        "universe": "SPY, ACWX, AGG (gated by BIL)",
        "source": "Antonacci (2014) 'Dual Momentum Investing' — GEM",
    })


if __name__ == "__main__":
    main()
