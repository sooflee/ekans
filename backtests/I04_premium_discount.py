"""
I04 Premium/Discount bias (ICT).

Setup: classic RSI(2) Connors mean-reversion (buy when RSI(2)<5; exit when close>5d SMA),
but only when SPY trades in the DISCOUNT half of the last 60-day range (price < 50% midpoint).
Compare to RSI(2) unfiltered (same close>200d SMA Connors regime).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import compute_metrics, print_metrics, save_result, mark_failed
from _ohlcv import load_ohlcv


def rsi_wilder(close, n=2):
    chg = close.diff()
    up = chg.clip(lower=0)
    dn = -chg.clip(upper=0)
    # Wilder's smoothing via EMA with alpha=1/n
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def simulate(close, entry_mask, exit_mask):
    """Long-only state machine: when entry true, hold; exit when exit true."""
    pos = pd.Series(0.0, index=close.index)
    holding = False
    for i in range(len(close)):
        if not holding and entry_mask.iloc[i]:
            holding = True
        elif holding and exit_mask.iloc[i]:
            holding = False
        pos.iloc[i] = 1.0 if holding else 0.0
    return pos


def main():
    try:
        ohlcv = load_ohlcv("SPY", start="2000-01-01")
    except Exception as e:
        return mark_failed("I04_premium_discount", f"data load failed: {e}")

    close = ohlcv["close"]
    rets = close.pct_change()
    sma200 = close.rolling(200).mean()
    sma5 = close.rolling(5).mean()
    high60 = ohlcv["high"].rolling(60).max()
    low60 = ohlcv["low"].rolling(60).min()
    mid60 = 0.5 * (high60 + low60)
    rsi2 = rsi_wilder(close, 2)

    discount = close < mid60

    # Unfiltered Connors RSI(2)
    entry_u = (close > sma200) & (rsi2 < 5)
    exit_u = close > sma5
    pos_u = simulate(close, entry_u, exit_u).shift(1).fillna(0)
    pnl_u = pos_u * rets

    # Discount-only
    entry_d = entry_u & discount
    pos_d = simulate(close, entry_d, exit_u).shift(1).fillna(0)
    pnl_d = pos_d * rets

    m_u = compute_metrics(pnl_u.dropna(), benchmark=rets, name="I04 RSI(2) unfiltered")
    m_d = compute_metrics(pnl_d.dropna(), benchmark=rets, name="I04 RSI(2) + Discount filter")
    print_metrics(m_u)
    print_metrics(m_d)

    save_result("I04_premium_discount", m_d, extra={
        "status": "ok",
        "rule": "Connors RSI(2)<5 long-only with close>200dSMA; require price<50% of last 60d range. "
                "Exit on close>5d SMA. Compare to unfiltered.",
        "universe": "SPY daily",
        "unfiltered_metrics": m_u,
        "n_entries_filtered": int(((entry_d) & (~entry_d.shift(1).fillna(False))).sum()),
        "n_entries_unfiltered": int(((entry_u) & (~entry_u.shift(1).fillna(False))).sum()),
        "source": "ICT premium/discount + Connors RSI(2)",
    })


if __name__ == "__main__":
    main()
