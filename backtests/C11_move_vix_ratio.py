"""
C11 MOVE/VIX ratio
^MOVE on yfinance vs ^VIX. When ratio > 6 sustained 5+ days -> reduce equity (cash 20 days);
when < 4 -> long equity 20 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        move = load_prices(["^MOVE"], start="2003-01-01").iloc[:, 0].rename("MOVE")
        vix = load_prices(["^VIX"], start="2003-01-01").iloc[:, 0].rename("VIX")
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("C11_move_vix_ratio", f"data load failed: {e}")

    df = pd.concat([move, vix, spy], axis=1).dropna()
    if df.empty:
        return mark_failed("C11_move_vix_ratio", "no overlap MOVE/VIX")
    rets = df["SPY"].pct_change()

    ratio = df["MOVE"] / df["VIX"]
    hi_trig = (ratio > 6).rolling(5).sum() == 5
    lo_trig = ratio < 4

    pos = pd.Series(1.0, index=df.index)  # default long
    rem_flat = 0
    rem_long = 0
    for i in range(len(df)):
        if hi_trig.iloc[i]:
            rem_flat = 20
        if lo_trig.iloc[i]:
            rem_long = 20
        if rem_flat > 0:
            pos.iloc[i] = 0.0
            rem_flat -= 1
            if rem_long > 0:
                rem_long -= 1
        elif rem_long > 0:
            pos.iloc[i] = 1.0
            rem_long -= 1
        else:
            pos.iloc[i] = 0.0  # neutral when no signal: keep cash to make signal informative

    # alternative: default long is more natural. Re-do with default long:
    pos = pd.Series(1.0, index=df.index)
    rem_flat = 0
    for i in range(len(df)):
        if hi_trig.iloc[i]:
            rem_flat = 20
        if rem_flat > 0:
            pos.iloc[i] = 0.0
            rem_flat -= 1
        else:
            pos.iloc[i] = 1.0  # always long unless under "flat" override

    pnl = (pos.shift(1) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="C11 MOVE/VIX ratio")
    print_metrics(m)
    save_result("C11_move_vix_ratio", m, extra={
        "status": "ok",
        "rule": "Default long SPY; when MOVE/VIX>6 for 5d -> cash 20d.  (Sub-rule: ratio<4 -> long 20d, dominated by default-long stance.)",
        "data_source": "yfinance ^MOVE, ^VIX",
    })


if __name__ == "__main__":
    main()
