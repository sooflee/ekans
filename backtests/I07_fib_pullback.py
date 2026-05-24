"""
I07 0.618 Fibonacci pullback.

Mechanical detection at bar t:
- Look back 50 bars; swing_low = min(low[t-50..t]), swing_high = max(high[t-50..t]).
  Use the simpler "highest high / lowest low" of the 50-bar window as the swing.
- Require swing_low_idx < swing_high_idx (uptrend swing).
- 50d MA rising at t (sma50[t] > sma50[t-5]).
- Pullback zone: price retraced to 50%-61.8% of (swing_high - swing_low).
  i.e. low[t] <= swing_high - 0.5 * range  AND  low[t] >= swing_high - 0.618 * range.
- Today is a bullish engulfing: close[t] > open[t] AND close[t] > open[t-1] AND open[t] < close[t-1]
  AND prev candle red (close[t-1] < open[t-1]).
- Enter next open. Stop = swing_low. Target = 1.618 extension from swing_low through swing_high:
  swing_high + 0.618 * range.
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
        return mark_failed("I07_fib_pullback", f"data load failed: {e}")

    o = ohlcv["open"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = ohlcv["close"].values
    idx = ohlcv.index
    n = len(ohlcv)
    sma50 = pd.Series(c, index=idx).rolling(50).mean().values

    trades = []
    last_exit = -1
    for t in range(55, n - 1):
        if t <= last_exit:
            continue
        if np.isnan(sma50[t]) or np.isnan(sma50[t - 5]) or sma50[t] <= sma50[t - 5]:
            continue
        window_h = h[t - 50:t + 1]
        window_l = l[t - 50:t + 1]
        sh_i = int(np.argmax(window_h))
        sl_i = int(np.argmin(window_l))
        if sl_i >= sh_i:
            continue
        swing_high = window_h[sh_i]
        swing_low = window_l[sl_i]
        rng = swing_high - swing_low
        if rng <= 0:
            continue
        fib_50 = swing_high - 0.5 * rng
        fib_618 = swing_high - 0.618 * rng
        if not (fib_618 <= l[t] <= fib_50):
            continue
        # bullish engulfing
        if not (c[t] > o[t] and c[t - 1] < o[t - 1] and c[t] > o[t - 1] and o[t] < c[t - 1]):
            continue
        # Enter next open
        k = t + 1
        if k >= n:
            break
        entry = o[k]
        stop = swing_low
        tgt = swing_high + 0.618 * rng
        if entry <= stop:
            continue
        exit_px = None
        exit_idx = None
        for m in range(k, min(k + 60, n)):
            if l[m] <= stop:
                exit_px = stop
                exit_idx = m
                break
            if h[m] >= tgt:
                exit_px = tgt
                exit_idx = m
                break
        if exit_px is None:
            exit_idx = min(k + 60 - 1, n - 1)
            exit_px = c[exit_idx]
        ret = exit_px / entry - 1.0
        trades.append({
            "signal_date": idx[t],
            "entry_date": idx[k],
            "exit_date": idx[exit_idx],
            "entry": float(entry),
            "exit": float(exit_px),
            "stop": float(stop),
            "target": float(tgt),
            "ret": float(ret),
            "hold_days": int(exit_idx - k + 1),
        })
        last_exit = exit_idx

    if not trades:
        return mark_failed("I07_fib_pullback", "no trades")

    tr = pd.DataFrame(trades)
    pnl = pd.Series(0.0, index=idx)
    for _, t in tr.iterrows():
        pnl.loc[t["exit_date"]] += t["ret"]
    pnl = pnl.loc[tr["entry_date"].min():]

    rets = tr["ret"].values
    mean = float(rets.mean())
    std = float(rets.std(ddof=1))
    tstat = mean / (std / np.sqrt(len(rets))) if std > 0 else 0.0
    hit = float((rets > 0).mean())

    m = compute_metrics(pnl.dropna(), name="I07 Fib 0.618 pullback")
    print_metrics(m)
    save_result("I07_fib_pullback", m, extra={
        "status": "ok",
        "rule": "50-bar swing (low before high); 50d SMA rising; pullback into 0.5-0.618 zone; "
                "bullish engulfing candle. Enter next open; stop=swing low; target=1.618 ext.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "Fibonacci retracement / retail TA",
    })


if __name__ == "__main__":
    main()
