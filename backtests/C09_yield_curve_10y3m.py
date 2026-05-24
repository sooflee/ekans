"""
C09 3m10y inversion
Similar to C08 but use T10Y3M; Estrella-Mishkin threshold:
when T10Y3M < 0 for 30 consecutive days, reduce equity.
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
        t10y3m = load_fred("T10Y3M", start="1990-01-01").iloc[:, 0].rename("T10Y3M")
        spy = load_prices(["SPY"], start="1993-01-29").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("C09_yield_curve_10y3m", f"data load failed: {e}")

    t10y3m = t10y3m.reindex(spy.index, method="ffill")
    df = pd.concat([spy, t10y3m], axis=1).dropna()

    spread = df["T10Y3M"]
    # 30 consecutive trading days < 0
    neg = (spread < 0).astype(int)
    rolling_neg = neg.rolling(30).sum()
    triggered = rolling_neg >= 30

    pos = pd.Series(1.0, index=df.index)
    state_reduced = False
    for i in range(len(df)):
        if triggered.iloc[i] and not state_reduced:
            state_reduced = True
        elif state_reduced and spread.iloc[i] > 0.5:
            state_reduced = False
        if state_reduced:
            pos.iloc[i] = 0.5

    rets = df["SPY"].pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C09 Yield curve (10Y-3M) inversion timer")
    print_metrics(m)
    save_result("C09_yield_curve_10y3m", m, extra={
        "status": "ok",
        "rule": "30 consecutive days T10Y3M<0 -> reduce SPY to 50% until spread >+0.5.",
        "universe": "SPY (gated by T10Y3M)",
        "source": "Estrella-Mishkin (1996)",
    })


if __name__ == "__main__":
    main()
