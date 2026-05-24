"""
M4 ETH/BTC ratio mean reversion to 200d MA.
Long ETH-USD when ratio closes above 200d MA after >=60 consecutive days below.
Hold until cross back below 200d MA, or max 180 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        px = load_prices(["ETH-USD", "BTC-USD"], start="2017-11-09")
    except Exception as e:
        return mark_failed("M4_eth_btc_ratio", f"data load failed: {e}")

    px = px.dropna()
    if len(px) < 300:
        return mark_failed("M4_eth_btc_ratio", f"insufficient data: {len(px)} rows")

    ratio = px["ETH-USD"] / px["BTC-USD"]
    ma200 = ratio.rolling(200).mean()
    df = pd.concat([px["ETH-USD"].rename("ETH"), ratio.rename("R"), ma200.rename("MA")], axis=1).dropna()

    below = df["R"] < df["MA"]
    # Consecutive days below
    run = below.astype(int).groupby((~below).cumsum()).cumsum()

    # Entry: today crosses above MA, and yesterday's run >= 60
    cross_up = (df["R"].shift(1) < df["MA"].shift(1)) & (df["R"] >= df["MA"])
    qual = (run.shift(1) >= 60) & cross_up

    pos = pd.Series(0.0, index=df.index)
    i = 0
    n_events = 0
    idx = df.index
    while i < len(idx):
        if qual.iloc[i]:
            n_events += 1
            entry = i
            max_exit = min(entry + 180, len(idx))
            exit_i = max_exit
            for j in range(entry + 1, max_exit):
                if df["R"].iloc[j] < df["MA"].iloc[j]:
                    exit_i = j
                    break
            pos.iloc[entry:exit_i] = 1.0
            i = exit_i
        else:
            i += 1

    rets = df["ETH"].pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M4 ETH/BTC ratio cross above 200d after >=60d below → long ETH")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M4_eth_btc_ratio", m, extra={
        "status": "ok",
        "rule": "Long ETH-USD when ETH/BTC ratio crosses above 200d MA after >=60 consecutive days below; exit on cross back below MA or 180d max.",
        "mechanism": "ETH/BTC regime change captures alt-coin risk appetite shifts; bottoming pattern after extended underperformance.",
        "source": "yfinance ETH-USD / BTC-USD; alt-season heuristic.",
    })


if __name__ == "__main__":
    main()
