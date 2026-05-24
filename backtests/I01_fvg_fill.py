"""
I01 ICT Fair Value Gap fill (event study).

Rule (mechanical):
- Bullish FVG formed at bar t (using 3 bars t-2, t-1, t):
    low[t] > high[t-2]  (a price gap exists between high[t-2] and low[t])
  Zone = [high[t-2], low[t]].
- For up to 20 days after t, if any future bar's low <= low[t] AND its close enters [high[t-2], low[t]],
  go LONG at next open. Stop = FVG low (= high[t-2]) minus a small buffer; we use stop = high[t-2].
  Target = prior 20-day high (of SPY's close before the FVG was formed).
- Exit when EITHER target hit (intraday high >= target) OR stop hit (intraday low <= stop) OR 20 trading
  days elapse from entry (whichever first).
- We treat returns as the trade's (exit_price/entry_price - 1) accrued on the exit day, and otherwise zero.
- Position size = 1 (cash equity).

Reported as an event-study: average per-trade return, hit rate, t-stat.
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
        return mark_failed("I01_fvg_fill", f"data load failed: {e}")

    h = ohlcv["high"].values
    l = ohlcv["low"].values
    c = ohlcv["close"].values
    o = ohlcv["open"].values
    idx = ohlcv.index

    prior20_high = ohlcv["high"].rolling(20).max().shift(1).values

    trades = []
    n = len(ohlcv)
    i = 2
    while i < n - 1:
        # Bullish FVG at bar i (uses bars i-2, i-1, i)
        if l[i] > h[i - 2]:
            zone_lo = h[i - 2]
            zone_hi = l[i]
            target = prior20_high[i]  # prior 20-day high (excluding today)
            if np.isnan(target):
                i += 1
                continue
            # Pullback window: next 20 bars
            entered = False
            for j in range(i + 1, min(i + 21, n)):
                # Pullback: today's range touches the zone AND close inside the zone
                if l[j] <= zone_hi and c[j] >= zone_lo and c[j] <= zone_hi:
                    # Enter next open
                    k = j + 1
                    if k >= n:
                        break
                    entry = o[k]
                    stop = zone_lo
                    # Walk forward up to 20 days
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
                        "fvg_date": idx[i],
                        "entry_date": idx[k],
                        "exit_date": idx[exit_idx],
                        "entry": float(entry),
                        "exit": float(exit_px),
                        "target": float(target),
                        "stop": float(stop),
                        "ret": float(ret),
                        "hold_days": int(exit_idx - k + 1),
                    })
                    entered = True
                    # skip past entry to avoid stacking on same FVG
                    i = exit_idx
                    break
            if not entered:
                pass
        i += 1

    if not trades:
        return mark_failed("I01_fvg_fill", "no trades triggered")

    tr = pd.DataFrame(trades)
    # Build a daily PnL series: each trade's return assigned to the exit date.
    pnl = pd.Series(0.0, index=idx)
    for _, t in tr.iterrows():
        pnl.loc[t["exit_date"]] += t["ret"]

    # Trim to first trade onward
    first = tr["entry_date"].min()
    pnl = pnl.loc[first:]

    # Per-trade stats (used as reported summary)
    rets = tr["ret"].values
    mean = float(rets.mean())
    std = float(rets.std(ddof=1))
    tstat = mean / (std / np.sqrt(len(rets))) if std > 0 else 0.0
    hit = float((rets > 0).mean())

    # Series metrics for record (compounded at exit dates)
    m = compute_metrics(pnl.dropna(), name="I01 FVG fill")
    print_metrics(m)

    extra = {
        "status": "ok",
        "rule": "Bullish FVG (low[t]>high[t-2]); enter next open after a close back into zone; "
                "stop=zone low; target=prior-20d high; max-hold 20 days.",
        "universe": "SPY daily",
        "n_trades": len(tr),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "avg_hold_days": float(tr["hold_days"].mean()),
        "source": "ICT (Inner Circle Trader) FVG concept; retail TA",
        "notes": "Event-study; no compounding across overlapping trades.",
    }
    save_result("I01_fvg_fill", m, extra=extra)


if __name__ == "__main__":
    main()
