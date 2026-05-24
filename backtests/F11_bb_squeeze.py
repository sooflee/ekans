"""
F11 Bollinger Band squeeze breakout.

Rule (mechanical):
- BB(20,2): SMA20 +/- 2*stdev20. Width = (upper-lower)/sma20.
- "6-month low" of width = today's width == min(width[t-126..t]).
- After such a squeeze, wait for the FIRST daily close outside the bands (upper or lower) within the
  next 20 trading days.
  - If close>upper: go LONG at next open; stop = lower band on entry day; target = entry + 1 * prior
    squeeze range, where squeeze range = (upper - lower) on the squeeze date.
  - If close<lower: go SHORT at next open; stop = upper band; target = entry - 1 * squeeze range.
- Max hold 40 days.
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
        return mark_failed("F11_bb_squeeze", f"data load failed: {e}")

    close = ohlcv["close"]
    o = ohlcv["open"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = close.values
    idx = ohlcv.index
    n = len(ohlcv)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = (sma20 + 2 * std20).values
    lower = (sma20 - 2 * std20).values
    width = ((upper - lower) / sma20.values)
    # 6-month = 126 trading days
    width_min = pd.Series(width, index=idx).rolling(126).min().values
    is_squeeze = (width == width_min) & ~np.isnan(width)

    trades = []
    last_exit = -1
    for t in range(126, n - 1):
        if t <= last_exit:
            continue
        if not is_squeeze[t]:
            continue
        squeeze_range = upper[t] - lower[t]
        # Wait next 20 days for close outside bands
        triggered = False
        for j in range(t + 1, min(t + 21, n)):
            if c[j] > upper[j]:
                direction = 1
                k = j + 1
                if k >= n:
                    break
                entry = o[k]
                stop = lower[j]
                tgt = entry + squeeze_range
                triggered = True
                break
            elif c[j] < lower[j]:
                direction = -1
                k = j + 1
                if k >= n:
                    break
                entry = o[k]
                stop = upper[j]
                tgt = entry - squeeze_range
                triggered = True
                break
        if not triggered:
            continue
        exit_px = None
        exit_idx = None
        for m in range(k, min(k + 40, n)):
            if direction == 1:
                if l[m] <= stop:
                    exit_px = stop
                    exit_idx = m
                    break
                if h[m] >= tgt:
                    exit_px = tgt
                    exit_idx = m
                    break
            else:
                if h[m] >= stop:
                    exit_px = stop
                    exit_idx = m
                    break
                if l[m] <= tgt:
                    exit_px = tgt
                    exit_idx = m
                    break
        if exit_px is None:
            exit_idx = min(k + 40 - 1, n - 1)
            exit_px = c[exit_idx]
        if direction == 1:
            ret = exit_px / entry - 1.0
        else:
            ret = (entry - exit_px) / entry
        trades.append({
            "squeeze_date": idx[t],
            "breakout_date": idx[k - 1],
            "entry_date": idx[k],
            "exit_date": idx[exit_idx],
            "direction": direction,
            "entry": float(entry),
            "exit": float(exit_px),
            "ret": float(ret),
            "hold_days": int(exit_idx - k + 1),
        })
        last_exit = exit_idx

    if not trades:
        return mark_failed("F11_bb_squeeze", "no trades")

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

    m = compute_metrics(pnl.dropna(), name="F11 Bollinger Squeeze breakout")
    print_metrics(m)
    save_result("F11_bb_squeeze", m, extra={
        "status": "ok",
        "rule": "BB(20,2) width at 126-day low; wait <=20d for close outside bands; enter next open "
                "in breakout direction; stop=opposite band; target=1x prior squeeze range; max 40d.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "n_long": int((tr["direction"] == 1).sum()),
        "n_short": int((tr["direction"] == -1).sum()),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "John Bollinger, Bollinger Band squeeze",
    })


if __name__ == "__main__":
    main()
