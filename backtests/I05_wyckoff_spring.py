"""
I05 Wyckoff Spring.

Rule (mechanical):
- 30-day range (prior 30 bars, excluding today): rng_lo = min(low[t-30..t-1]), rng_hi = max(high[t-30..t-1]).
- At bar t: low[t] < rng_lo AND close[t] > rng_lo AND volume[t] > 1.5 * 20d avg vol.
- Enter LONG at next open. Stop = today's low (spring low).
- Target = rng_hi. Also trail with 20-day MA: exit if close < 20d SMA.
- Max hold 60 trading days.
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
        return mark_failed("I05_wyckoff_spring", f"data load failed: {e}")

    o = ohlcv["open"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = ohlcv["close"].values
    v = ohlcv["volume"].values
    idx = ohlcv.index
    n = len(ohlcv)

    rng_lo = ohlcv["low"].rolling(30).min().shift(1).values
    rng_hi = ohlcv["high"].rolling(30).max().shift(1).values
    avg_vol = ohlcv["volume"].rolling(20).mean().shift(1).values
    sma20 = ohlcv["close"].rolling(20).mean().values

    trades = []
    i = 30
    while i < n - 1:
        if not (np.isnan(rng_lo[i]) or np.isnan(avg_vol[i])):
            if l[i] < rng_lo[i] and c[i] > rng_lo[i] and v[i] > 1.5 * avg_vol[i]:
                k = i + 1
                if k >= n:
                    break
                entry = o[k]
                stop = l[i]
                target = rng_hi[i]
                exit_px = None
                exit_idx = None
                for m in range(k, min(k + 60, n)):
                    if l[m] <= stop:
                        exit_px = stop
                        exit_idx = m
                        break
                    if h[m] >= target:
                        exit_px = target
                        exit_idx = m
                        break
                    # trail with 20-day MA, evaluate at close
                    if c[m] < sma20[m]:
                        exit_px = c[m]
                        exit_idx = m
                        break
                if exit_px is None:
                    exit_idx = min(k + 60 - 1, n - 1)
                    exit_px = c[exit_idx]
                ret = exit_px / entry - 1.0
                trades.append({
                    "spring_date": idx[i],
                    "entry_date": idx[k],
                    "exit_date": idx[exit_idx],
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop": float(stop),
                    "target": float(target),
                    "ret": float(ret),
                    "hold_days": int(exit_idx - k + 1),
                })
                i = exit_idx
        i += 1

    if not trades:
        return mark_failed("I05_wyckoff_spring", "no trades")

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

    m = compute_metrics(pnl.dropna(), name="I05 Wyckoff Spring")
    print_metrics(m)
    save_result("I05_wyckoff_spring", m, extra={
        "status": "ok",
        "rule": "30-day range; if low<rng_lo and close>rng_lo and vol>1.5x20davg, enter next open; "
                "stop=spring low; target=rng high; trail with 20d MA; max hold 60d.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "Wyckoff Spring (retail TA)",
    })


if __name__ == "__main__":
    main()
