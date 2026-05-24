"""
F08 RSI(2) Mean Reversion (Connors).

Rule:
- Buy SPY at close when close > 200d SMA AND RSI(2) < 5.
- Exit at close when close > 5d SMA.
- Pos applied to NEXT day's return (no look-ahead).
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
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def main():
    try:
        ohlcv = load_ohlcv("SPY", start="2000-01-01")
    except Exception as e:
        return mark_failed("F08_rsi2_connors", f"data load failed: {e}")

    close = ohlcv["close"]
    rets = close.pct_change()
    sma200 = close.rolling(200).mean()
    sma5 = close.rolling(5).mean()
    rsi2 = rsi_wilder(close, 2)

    pos = pd.Series(0.0, index=close.index)
    holding = False
    n_entries = 0
    for i in range(len(close)):
        if not holding:
            if (close.iloc[i] > sma200.iloc[i]) and (rsi2.iloc[i] < 5):
                holding = True
                n_entries += 1
        else:
            if close.iloc[i] > sma5.iloc[i]:
                holding = False
        pos.iloc[i] = 1.0 if holding else 0.0

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="F08 Connors RSI(2)")
    print_metrics(m)
    save_result("F08_rsi2_connors", m, extra={
        "status": "ok",
        "rule": "Long SPY at close when close>200d SMA and RSI(2)<5; exit when close>5d SMA.",
        "universe": "SPY daily",
        "n_entries": int(n_entries),
        "source": "Connors & Alvarez 'Short Term Trading Strategies That Work'",
    })


if __name__ == "__main__":
    main()
