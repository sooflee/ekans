"""
F06 DeMark TD Sequential setup.

Rule:
- Buy setup completion: 9 consecutive bars where close[t] < close[t-4]. On bar of completion (the 9th),
  enter LONG at next open (we FADE the down move, i.e. buy a buy-setup). Hold 5 trading days.
- Sell setup completion: 9 consecutive bars where close[t] > close[t-4]. On the 9th bar, SHORT next
  open. Hold 5 trading days.
- Net daily PnL = long_pnl + short_pnl.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import compute_metrics, print_metrics, save_result, mark_failed
from _ohlcv import load_ohlcv


def main():
    try:
        ohlcv = load_ohlcv("SPY", start="2000-01-01")
    except Exception as e:
        return mark_failed("F06_td_sequential", f"data load failed: {e}")

    close = ohlcv["close"]
    rets = close.pct_change()
    idx = ohlcv.index
    n = len(ohlcv)

    # buy setup: 9 consecutive c[t]<c[t-4]
    cond_buy = (close < close.shift(4)).astype(int)
    cond_sell = (close > close.shift(4)).astype(int)

    # Count streaks
    buy_streak = pd.Series(0, index=idx)
    sell_streak = pd.Series(0, index=idx)
    bs = 0
    ss = 0
    for i in range(n):
        bs = bs + 1 if cond_buy.iloc[i] == 1 else 0
        ss = ss + 1 if cond_sell.iloc[i] == 1 else 0
        buy_streak.iloc[i] = bs
        sell_streak.iloc[i] = ss

    pos = pd.Series(0.0, index=idx)
    n_buy = 0
    n_sell = 0
    holds_remaining = 0
    direction = 0
    for i in range(n - 1):
        if holds_remaining > 0:
            pos.iloc[i] = direction
            holds_remaining -= 1
            continue
        # trigger at i with hold starting i+1
        if buy_streak.iloc[i] >= 9:
            direction = 1.0
            holds_remaining = 5
            n_buy += 1
        elif sell_streak.iloc[i] >= 9:
            direction = -1.0
            holds_remaining = 5
            n_sell += 1
        # else stay flat (already pos.iloc[i]=0)

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="F06 TD Sequential 9-bar setup fade")
    print_metrics(m)
    save_result("F06_td_sequential", m, extra={
        "status": "ok",
        "rule": "9 closes < close[-4] = buy setup -> long next open, hold 5d. "
                "9 closes > close[-4] = sell setup -> short next open, hold 5d. Fade direction.",
        "universe": "SPY daily",
        "n_buy_setups": int(n_buy),
        "n_sell_setups": int(n_sell),
        "source": "Tom DeMark TD Sequential (setup phase)",
    })


if __name__ == "__main__":
    main()
