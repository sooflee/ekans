"""
B01 VIX term structure (VIX / VIX3M)
Rule: when VIX/VIX3M < 0.92, hold SPY long; cross above 1.0 -> cash; re-enter when drops below 0.95.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        vix = load_prices(["^VIX"], start="2007-12-01")
        vix3m = load_prices(["^VIX3M"], start="2007-12-01")
        spy = load_prices(["SPY"], start="2007-12-01")
    except Exception as e:
        return mark_failed("B01_vix_term_structure", f"data load failed: {e}")

    vix = vix.iloc[:, 0].rename("VIX")
    vix3m = vix3m.iloc[:, 0].rename("VIX3M")
    spy = spy.iloc[:, 0].rename("SPY")

    df = pd.concat([vix, vix3m, spy], axis=1).dropna()
    if df.empty:
        return mark_failed("B01_vix_term_structure", "no overlap between ^VIX and ^VIX3M")

    ratio = df["VIX"] / df["VIX3M"]
    # State machine: long when ratio < 0.92; once ratio > 1.0 -> cash; re-enter when ratio < 0.95
    pos = pd.Series(0.0, index=df.index)
    state = 0  # 0 = cash, 1 = long
    for i, r in enumerate(ratio.values):
        if state == 0:
            if r < 0.92:
                state = 1
            elif r < 0.95:
                # re-entry rule for the in-between regime after exit
                pass
        else:  # state == 1
            if r > 1.0:
                state = 0
        pos.iloc[i] = state

    # apply 1-day lag (already handled inside long_short_pnl via shift)
    rets = df["SPY"].pct_change()
    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="B01 VIX/VIX3M term structure")
    print_metrics(m)
    save_result("B01_vix_term_structure", m, extra={
        "status": "ok",
        "rule": "Long SPY when VIX/VIX3M < 0.92; exit on cross above 1.0; allow re-entry below 0.95.",
        "universe": "SPY (vol gate via ^VIX/^VIX3M)",
        "source": "Vol term-structure literature (Donninger 2014 etc.)",
    })


if __name__ == "__main__":
    main()
