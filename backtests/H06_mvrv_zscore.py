"""
H06 MVRV Z-score (proxy).

True MVRV = (market cap - realized cap) / sigma(market cap). Realized cap is
behind a paywall on Glassnode/CoinMetrics free tier. We use the lightweight
proxy explicitly allowed by the brief: z-score of (BTC / 200d MA).

z = (price/MA200 - mean) / std, where mean/std are computed over a long
look-back (full history available, expanding window) so that we don't introduce
forward-look bias.

Rule: long BTC when z < 0 (price below long-run norm vs trend); flat when z > 5.
Otherwise hold previous state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["BTC-USD"], start="2014-01-01").iloc[:, 0]
    ma = px.rolling(200).mean()
    ratio = (px / ma).dropna()

    # expanding-window z to avoid look-ahead
    exp_mean = ratio.expanding(252).mean()
    exp_std = ratio.expanding(252).std()
    z = (ratio - exp_mean) / exp_std

    # state machine
    state = 0
    pos = pd.Series(0.0, index=ratio.index)
    for d, v in z.items():
        if pd.isna(v):
            pos.loc[d] = state
            continue
        if v < 0:
            state = 1
        elif v > 5:
            state = 0
        pos.loc[d] = state

    rets = px.pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[z.dropna().index[0]:]

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index], name="H06 MVRV z-score proxy")
    print_metrics(m)
    save_result("H06_mvrv_zscore", m, extra={
        "status": "ok",
        "rule": "z-score of (BTC / 200d MA) over expanding window: long when z<0; flat when z>5.",
        "data_source": "BTC-USD yfinance; proxy used because true realized-cap series is paywalled.",
        "note": "Brief explicitly permits this proxy as a fallback to Glassnode-paywalled realized-cap series.",
    })


if __name__ == "__main__":
    main()
