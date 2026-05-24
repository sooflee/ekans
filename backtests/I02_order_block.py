"""
I02 ICT Order Block retest.

Rule (mechanical):
- Identify a "displacement" up-candle at bar t: close[t] - open[t-1] > 1.5 * ATR(14)[t-1] above the
  prior down-candle's range, more concretely: close[t] > high[t-1] + 1.5*ATR14[t-1] AND
  candle t-1 is a down-candle (close[t-1] < open[t-1]).
- The down-candle at t-1 is the bullish order block; its range = [low[t-1], high[t-1]].
- In the next 20 trading days, if any bar's low touches the OB high (i.e. retests the upper edge),
  enter LONG at next open. Stop = OB low. Target = prior 20-day high (computed BEFORE the OB date).
- Exit on stop hit, target hit, or 20 days, whichever first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import compute_metrics, print_metrics, save_result, mark_failed
from _ohlcv import load_ohlcv, atr


def main():
    try:
        ohlcv = load_ohlcv("SPY", start="2000-01-01")
    except Exception as e:
        return mark_failed("I02_order_block", f"data load failed: {e}")

    o = ohlcv["open"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = ohlcv["close"].values
    a14 = atr(ohlcv, 14).values
    prior20_high = ohlcv["high"].rolling(20).max().shift(1).values

    idx = ohlcv.index
    n = len(ohlcv)

    trades = []
    i = 2
    while i < n - 1:
        # Down-candle at i-1; displacement up-candle at i
        if c[i - 1] < o[i - 1] and not np.isnan(a14[i - 1]):
            if c[i] > h[i - 1] + 1.5 * a14[i - 1]:
                ob_low = l[i - 1]
                ob_high = h[i - 1]
                target = prior20_high[i - 1]
                if np.isnan(target):
                    i += 1
                    continue
                # Retest within next 20 days
                for j in range(i + 1, min(i + 21, n)):
                    if l[j] <= ob_high:
                        k = j + 1
                        if k >= n:
                            break
                        entry = o[k]
                        stop = ob_low
                        if entry <= stop:
                            break
                        exit_px = None
                        exit_idx = None
                        for m in range(k, min(k + 20, n)):
                            if l[m] <= stop:
                                exit_px = stop
                                exit_idx = m
                                break
                            if h[m] >= target:
                                exit_px = target
                                exit_idx = m
                                break
                        if exit_px is None:
                            exit_idx = min(k + 20 - 1, n - 1)
                            exit_px = c[exit_idx]
                        ret = exit_px / entry - 1.0
                        trades.append({
                            "ob_date": idx[i - 1],
                            "entry_date": idx[k],
                            "exit_date": idx[exit_idx],
                            "entry": float(entry),
                            "exit": float(exit_px),
                            "target": float(target),
                            "stop": float(stop),
                            "ret": float(ret),
                            "hold_days": int(exit_idx - k + 1),
                        })
                        i = exit_idx
                        break
        i += 1

    if not trades:
        return mark_failed("I02_order_block", "no trades triggered")

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

    m = compute_metrics(pnl.dropna(), name="I02 ICT Order Block retest")
    print_metrics(m)

    save_result("I02_order_block", m, extra={
        "status": "ok",
        "rule": "Down-candle at t-1 followed by close[t]>high[t-1]+1.5*ATR14 = bullish OB; "
                "enter next open after retest of OB high within 20d; stop=OB low; target=prior-20d high.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "ICT (Inner Circle Trader) order-block concept; retail TA",
    })


if __name__ == "__main__":
    main()
