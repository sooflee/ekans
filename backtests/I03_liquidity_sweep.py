"""
I03 Liquidity sweep reversal (failed breakout fade).

Rule (mechanical):
- prior20_high = max(high[t-20..t-1]); prior20_low = min(low[t-20..t-1]).
- If high[t] > prior20_high AND close[t] < prior20_high:
    short at next open[t+1]; cover when intraday low <= midpoint(prior20_low, prior20_high)
    OR after 20 trading days (whichever first). Use t+1's prior20_low/high (i.e. computed at decision
    time t). Stop loss: cover if close goes back above today's high.
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
        return mark_failed("I03_liquidity_sweep", f"data load failed: {e}")

    o = ohlcv["open"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = ohlcv["close"].values
    idx = ohlcv.index
    n = len(ohlcv)

    p20h = ohlcv["high"].rolling(20).max().shift(1).values
    p20l = ohlcv["low"].rolling(20).min().shift(1).values

    trades = []
    i = 20
    while i < n - 1:
        if not (np.isnan(p20h[i]) or np.isnan(p20l[i])):
            if h[i] > p20h[i] and c[i] < p20h[i]:
                # Trigger: short next open
                k = i + 1
                if k >= n:
                    break
                entry = o[k]
                midpoint = 0.5 * (p20h[i] + p20l[i])
                stop = h[i]  # if intraday > today's high, cover
                exit_px = None
                exit_idx = None
                for m in range(k, min(k + 20, n)):
                    # cover at midpoint if hit intraday
                    if l[m] <= midpoint:
                        exit_px = midpoint
                        exit_idx = m
                        break
                    if h[m] >= stop:
                        exit_px = stop
                        exit_idx = m
                        break
                if exit_px is None:
                    exit_idx = min(k + 20 - 1, n - 1)
                    exit_px = c[exit_idx]
                # Short return = (entry - exit)/entry
                ret = (entry - exit_px) / entry
                trades.append({
                    "sweep_date": idx[i],
                    "entry_date": idx[k],
                    "exit_date": idx[exit_idx],
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "ret": float(ret),
                    "hold_days": int(exit_idx - k + 1),
                })
                i = exit_idx
        i += 1

    if not trades:
        return mark_failed("I03_liquidity_sweep", "no trades")

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

    m = compute_metrics(pnl.dropna(), name="I03 Liquidity sweep reversal")
    print_metrics(m)
    save_result("I03_liquidity_sweep", m, extra={
        "status": "ok",
        "rule": "If high[t]>prior20_high but close[t]<prior20_high, short next open, cover at midpoint "
                "of prior 20-day range or after 20 days (stop if high above today's high).",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "ICT/retail TA 'liquidity sweep' fade",
    })


if __name__ == "__main__":
    main()
