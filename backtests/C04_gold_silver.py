"""
C04 Gold/Silver ratio
GC=F / SI=F (fallback GLD/SLV). Z-score 5y. Long silver / short gold when ratio > 90;
reverse when < 50. Exit at 70.
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
        gld = load_prices(["GLD"], start="2005-01-01").iloc[:, 0].rename("GLD")
        slv = load_prices(["SLV"], start="2006-04-01").iloc[:, 0].rename("SLV")
        gc = load_prices(["GC=F"], start="2000-01-01").iloc[:, 0].rename("GC")
        si = load_prices(["SI=F"], start="2000-01-01").iloc[:, 0].rename("SI")
    except Exception as e:
        return mark_failed("C04_gold_silver", f"data load failed: {e}")

    # Use spot futures for the ratio (better history), ETFs for P&L
    fut = pd.concat([gc, si], axis=1).dropna()
    ratio = fut["GC"] / fut["SI"]

    etf = pd.concat([gld, slv], axis=1).dropna()
    ratio_aligned = ratio.reindex(etf.index, method="ffill")

    df = etf.copy()
    df["ratio"] = ratio_aligned
    df = df.dropna()

    # State machine: position in [silver - gold] (long SLV / short GLD)
    # >90: long silver, short gold (+1)
    # <50: long gold, short silver (-1)
    # exit at 70 (back to flat)
    pos = pd.Series(0.0, index=df.index)
    state = 0
    for i, r in enumerate(df["ratio"].values):
        if state == 0:
            if r > 90:
                state = 1
            elif r < 50:
                state = -1
        elif state == 1:
            if r < 70:
                state = 0
        elif state == -1:
            if r > 70:
                state = 0
        pos.iloc[i] = state

    gld_ret = df["GLD"].pct_change()
    slv_ret = df["SLV"].pct_change()

    pnl = pos.shift(1) * (slv_ret - gld_ret)
    pnl = pnl.dropna()

    bench = gld_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C04 Gold/Silver ratio")
    print_metrics(m)
    save_result("C04_gold_silver", m, extra={
        "status": "ok",
        "rule": "Ratio = gold/silver. >90: long SLV, short GLD. <50: long GLD, short SLV. Exit at 70.",
        "universe": "GLD / SLV (long/short)",
        "source": "Trader folklore; gold/silver ratio mean reversion",
    })


if __name__ == "__main__":
    main()
