"""
H07 Puell Multiple.

Puell = (Daily USD value of newly issued BTC) / (365d MA of same).
We compute new issuance from halving epochs (144 blocks/day * subsidy),
multiplied by BTC-USD price.

Halving epochs (subsidy in BTC/block):
  pre 2012-11-28: 50
  pre 2016-07-09: 25
  pre 2020-05-11: 12.5
  pre 2024-04-19: 6.25
  >=2024-04-19:   3.125

Rule: long BTC when Puell < 0.5 (miner stress -> historical accumulation zone);
flat when Puell > 3.0. Hold previous state otherwise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result,
)


HALVING_DATES = [
    (pd.Timestamp("2009-01-03"), 50.0),
    (pd.Timestamp("2012-11-28"), 25.0),
    (pd.Timestamp("2016-07-09"), 12.5),
    (pd.Timestamp("2020-05-11"), 6.25),
    (pd.Timestamp("2024-04-19"), 3.125),
]


def subsidy_at(d):
    s = 50.0
    for h, new_s in HALVING_DATES:
        if d >= h:
            s = new_s
    return s


def main():
    px = load_prices(["BTC-USD"], start="2014-01-01").iloc[:, 0]

    subsidy = pd.Series([subsidy_at(d) for d in px.index], index=px.index)
    issuance_btc = 144 * subsidy  # BTC issued per day
    issuance_usd = issuance_btc * px

    sma365 = issuance_usd.rolling(365).mean()
    puell = (issuance_usd / sma365).dropna()

    state = 0
    pos = pd.Series(0.0, index=puell.index)
    for d, v in puell.items():
        if v < 0.5:
            state = 1
        elif v > 3.0:
            state = 0
        pos.loc[d] = state

    rets = px.pct_change()
    pnl = (pos.shift(1) * rets).dropna()

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index], name="H07 Puell Multiple")
    print_metrics(m)
    save_result("H07_puell_multiple", m, extra={
        "status": "ok",
        "rule": "Long when Puell<0.5; flat when Puell>3.0; hold otherwise.",
        "data_source": "BTC issuance from halving-epoch subsidy (144 blocks/day); price from yfinance.",
        "puell_min": float(puell.min()),
        "puell_max": float(puell.max()),
    })


if __name__ == "__main__":
    main()
