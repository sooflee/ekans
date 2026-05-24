"""
B02 VIX > 30 contrarian buy
When VIX closes > 30 having previously been < 20 within last 60 days, buy SPY at next open.
Exit when VIX closes < 20 OR after 60 trading days.
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
        vix = load_prices(["^VIX"], start="1993-01-01").iloc[:, 0].rename("VIX")
        spy = load_prices(["SPY"], start="1993-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("B02_vix_30_contrarian", f"data load failed: {e}")

    df = pd.concat([vix, spy], axis=1).dropna()
    rets = df["SPY"].pct_change()

    # was_below_20 in last 60d?
    below_20 = (df["VIX"] < 20).rolling(60).max().fillna(0).astype(bool)
    cross_above_30 = (df["VIX"] > 30) & (df["VIX"].shift(1) <= 30)
    entry_trigger = cross_above_30 & below_20

    pos = pd.Series(0.0, index=df.index)
    state = 0
    days_in = 0
    for i in range(len(df)):
        v = df["VIX"].iloc[i]
        if state == 0:
            if entry_trigger.iloc[i]:
                state = 1
                days_in = 0
        else:
            days_in += 1
            if v < 20 or days_in >= 60:
                state = 0
                days_in = 0
        pos.iloc[i] = state

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="B02 VIX>30 contrarian")
    print_metrics(m)
    save_result("B02_vix_30_contrarian", m, extra={
        "status": "ok",
        "rule": "Enter SPY long when VIX crosses >30 after being <20 within last 60d; exit when VIX<20 or after 60 days.",
        "universe": "SPY (signal: ^VIX)",
        "source": "Whaley fear-gauge / classic contrarian VIX literature",
    })


if __name__ == "__main__":
    main()
