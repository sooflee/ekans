"""
C02 Utilities/SPY ratio (Gayed)
4-week return XLU minus SPY. If XLU > SPY, hold cash next 4 weeks; else long SPY.
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
    try:
        xlu = load_prices(["XLU"], start="2000-01-01").iloc[:, 0].rename("XLU")
        spy = load_prices(["SPY"], start="2000-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("C02_utilities_spy", f"data load failed: {e}")

    df = pd.concat([xlu, spy], axis=1).dropna()
    look = 20  # 4 weeks ~ 20 trading days
    r_xlu = df["XLU"].pct_change(look)
    r_spy = df["SPY"].pct_change(look)

    # Signal: when XLU > SPY (defensive), go to cash; else long SPY
    raw_sig = (r_xlu <= r_spy).astype(float)  # 1 = long SPY

    # 4-week rebalance: only allow position changes on monthly (4-week) cadence
    monthly = raw_sig.resample("4W-FRI").last().reindex(df.index, method="ffill")

    spy_ret = df["SPY"].pct_change()
    pos = monthly.shift(1)
    pnl = (pos * spy_ret).dropna()

    bench = spy_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C02 Utilities/SPY (Gayed)")
    print_metrics(m)
    save_result("C02_utilities_spy", m, extra={
        "status": "ok",
        "rule": "4-wk return XLU vs SPY: if XLU>SPY hold cash, else long SPY. Rebalanced every 4 weeks.",
        "universe": "SPY (gated by XLU/SPY 4-wk relative return)",
        "source": "Gayed — Utilities relative performance",
    })


if __name__ == "__main__":
    main()
