"""
F10 MACD bullish divergence.

Rule (mechanical):
- MACD line = EMA12(close) - EMA26(close); Signal = EMA9(MACD); Histogram = MACD - Signal.
- On day t, if close[t] == min(close[t-20..t]) (20-day low) AND there exists a PRIOR 20-day-low
  date p (p<t) within last 100 trading days where MACD_hist[t] > MACD_hist[p] -> bullish divergence.
- Buy at next open. Exit at close when MACD < Signal (signal-line cross down).
- Max hold 60 days.
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
        return mark_failed("F10_macd_divergence", f"data load failed: {e}")

    close = ohlcv["close"]
    o = ohlcv["open"].values
    c = ohlcv["close"].values
    idx = ohlcv.index
    n = len(ohlcv)
    rets = close.pct_change()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = (macd - sig).values

    # 20-day low days (close)
    rolling_min = close.rolling(20).min().values
    is_20d_low = (c == rolling_min)

    # Find divergence entries
    trades = []
    last_exit = -1
    low_indices = [i for i in range(20, n) if is_20d_low[i]]
    low_set = set(low_indices)

    for t in low_indices:
        if t <= last_exit:
            continue
        # Find a prior 20-day-low within [t-100, t-1] with hist[t] > hist[p]
        diverged = False
        for p in range(max(20, t - 100), t):
            if p in low_set and not np.isnan(hist[p]) and not np.isnan(hist[t]):
                if hist[t] > hist[p]:
                    diverged = True
                    break
        if not diverged:
            continue
        k = t + 1
        if k >= n:
            break
        entry = o[k]
        # Exit when MACD < Signal at close
        exit_idx = None
        for m in range(k, min(k + 60, n)):
            if macd.iloc[m] < sig.iloc[m]:
                exit_idx = m
                break
        if exit_idx is None:
            exit_idx = min(k + 60 - 1, n - 1)
        exit_px = c[exit_idx]
        ret = exit_px / entry - 1.0
        trades.append({
            "signal_date": idx[t],
            "entry_date": idx[k],
            "exit_date": idx[exit_idx],
            "entry": float(entry),
            "exit": float(exit_px),
            "ret": float(ret),
            "hold_days": int(exit_idx - k + 1),
        })
        last_exit = exit_idx

    if not trades:
        return mark_failed("F10_macd_divergence", "no trades")

    tr = pd.DataFrame(trades)
    pnl = pd.Series(0.0, index=idx)
    for _, t in tr.iterrows():
        pnl.loc[t["exit_date"]] += t["ret"]
    pnl = pnl.loc[tr["entry_date"].min():]

    rets_arr = tr["ret"].values
    mean = float(rets_arr.mean())
    std = float(rets_arr.std(ddof=1))
    tstat = mean / (std / np.sqrt(len(rets_arr))) if std > 0 else 0.0
    hit = float((rets_arr > 0).mean())

    m = compute_metrics(pnl.dropna(), name="F10 MACD bullish divergence")
    print_metrics(m)
    save_result("F10_macd_divergence", m, extra={
        "status": "ok",
        "rule": "20-day close low AND MACD(12,26,9) hist higher than at a prior 20-day low within last "
                "100 days. Long next open; exit on MACD<Signal close; max hold 60d.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "Classic MACD divergence (Appel/Murphy)",
    })


if __name__ == "__main__":
    main()
