"""
N7 Coinbase exchange volume drop.
Coinbase exchange historical trading volume isn't exposed as a clean public time
series; use COIN equity daily share volume as proxy (yfinance), since COIN's own
trading activity correlates with crypto activity / exchange revenue. COIN started
trading Apr 14 2021.
Rule: short COIN 10d when 30d avg volume drops > 50% vs its prior 90d level, for
3 consecutive days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from harness import compute_metrics, print_metrics, save_result, mark_failed


def main():
    try:
        df = yf.download("COIN", start="2021-04-14", progress=False, auto_adjust=True)
        if df.empty:
            return mark_failed("N7_coinbase_volume", "yfinance returned empty for COIN")
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]["COIN"]
            vol = df["Volume"]["COIN"]
        else:
            close = df["Close"]
            vol = df["Volume"]
    except Exception as e:
        return mark_failed("N7_coinbase_volume", f"data load failed: {e}")

    v30 = vol.rolling(30).mean()
    v90 = vol.rolling(90).mean().shift(30)  # baseline before the 30d window
    ratio = v30 / v90
    drop = ratio < 0.5  # >50% drop
    sig = drop & drop.shift(1).fillna(False) & drop.shift(2).fillna(False)  # 3 consecutive
    trig_dates = drop.index[sig.fillna(False)]

    rets = close.pct_change()
    pos = pd.Series(0.0, index=rets.index)
    for d in trig_dates:
        loc = rets.index.searchsorted(d)
        for k in range(1, 11):
            if loc + k < len(pos):
                pos.iloc[loc + k] = -1.0  # short

    if pos.eq(0).all():
        return mark_failed(
            "N7_coinbase_volume",
            "no triggers in available COIN history",
            extra={"n_triggers": 0,
                   "rule": "Short COIN 10d when 30d-avg-volume / 90d-avg-volume < 0.5 for 3 consecutive days"},
        )

    pnl_full = (pos * rets).dropna()
    pnl = pnl_full.loc[pnl_full.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N7 COIN volume drop short")
    m["n_triggers"] = int(sig.fillna(False).sum())
    print(f"Trigger days: {m['n_triggers']}")
    print_metrics(m)
    save_result("N7_coinbase_volume", m, extra={
        "status": "ok",
        "rule": "Short COIN 10 sessions when 30d-avg COIN share-volume / prior-90d-avg < 0.5 for 3 consecutive days.",
        "mechanism": "Falling COIN activity proxies for crypto-exchange revenue compression -> earnings risk",
        "universe": "COIN",
        "source": "yfinance COIN share volume (proxy for Coinbase exchange BTC/USD volume)",
        "data_substitution": "Coinbase exchange historical trading volume not exposed as clean free time series; using COIN equity volume as proxy.",
    })


if __name__ == "__main__":
    main()
