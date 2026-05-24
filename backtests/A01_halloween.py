"""
A01 Halloween (Sell-in-May)
Long SPY from close of last trading day in October to close of last trading day in April;
cash otherwise. Compare to buy-and-hold SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]

    # Build positions: 1 if we're in the Nov–Apr window (inclusive of last Oct close)
    # Rule: position from close of last trading day in October => effectively long during
    # Nov, Dec, Jan, Feb, Mar, Apr. Cash May–Oct.
    idx = rets.index
    months = idx.month
    pos = pd.Series(0.0, index=idx)
    pos[(months >= 11) | (months <= 4)] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A01 Halloween")
    print_metrics(m)
    save_result("A01_halloween", m, extra={
        "status": "ok",
        "rule": "Long SPY in Nov-Apr; cash May-Oct (0% return for cash leg).",
        "universe": "SPY",
        "source": "Bouman & Jacobsen 2002",
    })


if __name__ == "__main__":
    main()
