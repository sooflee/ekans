"""
F09 Golden Cross / Death Cross.

Rule:
- Long SPY when 50d SMA > 200d SMA, flat otherwise. Compare to buy-and-hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import compute_metrics, print_metrics, save_result, load_prices, daily_returns


def main():
    px = load_prices(["SPY"], start="2000-01-01")["SPY"]
    rets = px.pct_change()
    s50 = px.rolling(50).mean()
    s200 = px.rolling(200).mean()
    pos = (s50 > s200).astype(float)
    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets.dropna(), name="F09 Golden Cross 50/200")
    print_metrics(m)
    n_flips = int(pos.diff().abs().sum())
    save_result("F09_golden_cross", m, extra={
        "status": "ok",
        "rule": "Long SPY when 50d SMA > 200d SMA, else flat.",
        "universe": "SPY daily",
        "n_signal_flips": n_flips,
        "source": "Classic moving-average crossover (golden/death cross)",
    })


if __name__ == "__main__":
    main()
