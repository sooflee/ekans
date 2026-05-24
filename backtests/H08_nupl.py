"""
H08 NUPL (Net Unrealized Profit/Loss) — proxy.

True NUPL = (market cap - realized cap) / market cap; realized cap is
paywalled. We use a proxy based on trailing 365d max drawdown / unrealized
profit relative to long-run price trend:

  proxy_NUPL_t = (P_t - MA365_t) / P_t.

This captures the same spirit — how far above the "cost basis trend" the
current price sits, scaled to current price. Range roughly [-0.6, +0.9] in
practice.

Classical NUPL zones (Greg Foss / glassnode lore):
  <0      = "Capitulation"
  0..0.25 = "Hope/Fear"
  0.25..0.5 = "Optimism/Anxiety"
  0.5..0.75 = "Belief/Denial"
  >0.75   = "Euphoria/Greed" (top zone)

Rule: long BTC when proxy < 0 (capitulation); flat when proxy > 0.6 (euphoria).
Hold previous state otherwise.
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
    ma = px.rolling(365).mean()
    proxy = ((px - ma) / px).dropna()

    state = 0
    pos = pd.Series(0.0, index=proxy.index)
    for d, v in proxy.items():
        if v < 0:
            state = 1
        elif v > 0.6:
            state = 0
        pos.loc[d] = state

    rets = px.pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[proxy.index[0]:]

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index], name="H08 NUPL proxy")
    print_metrics(m)
    save_result("H08_nupl", m, extra={
        "status": "ok",
        "rule": "Long when (P-MA365)/P < 0 (capitulation proxy); flat when > 0.6 (euphoria proxy).",
        "data_source": "BTC-USD yfinance; proxy used because realized-cap is paywalled.",
        "note": "Brief permits proxy approach.",
    })


if __name__ == "__main__":
    main()
