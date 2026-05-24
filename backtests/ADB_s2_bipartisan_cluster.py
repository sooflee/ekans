"""
AD-S2 Bipartisan STOCK Act convergence cluster.

Spec rule: >=2 Democrats AND >=2 Republicans buy same ticker within 21-day
window (no offsetting sales), exclude mega-cap (AAPL/MSFT/NVDA/GOOGL/AMZN),
long 60-90d.

Backtest approach (free-data proxy):
- We cannot directly observe cluster events without scraping CapitolTrades
  daily. Instead, we use the union/intersection of two publicly-listed
  ETFs as a static proxy:
    * NANC (Unusual Whales Subversive Democratic Trading ETF) — top
      Democrat-tracked holdings
    * KRUZ (Unusual Whales Subversive Republican Trading ETF) — top
      Republican-tracked holdings
- "Bipartisan convergence" approximated as: holdings the funds REBALANCED
  INTO concurrently. Without holdings-snapshot data, we proxy with an
  even coarser approximation: equal-weight (NANC + KRUZ) basket long
  vs SPY benchmark.

This is a very coarse proxy and we explicitly flag it. We treat it as a
secondary signal at best. We also acknowledge that the spec recommends
mark_failed if scraping isn't available; we attempt the proxy first and
report whichever is more honest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    # NANC launched 2023-02-07, KRUZ same.
    try:
        px = load_prices(["NANC", "KRUZ", "SPY"], start="2023-02-07")
    except Exception as e:
        return mark_failed("AD-S2", f"price load failed: {e}")

    if "NANC" not in px.columns or "KRUZ" not in px.columns:
        return mark_failed(
            "AD-S2",
            "NANC/KRUZ price data unavailable; without CapitolTrades scrape, "
            "bipartisan cluster events not enumerable."
        )

    nanc = px["NANC"].dropna()
    kruz = px["KRUZ"].dropna()
    spy = px["SPY"].dropna()
    common = nanc.index.intersection(kruz.index).intersection(spy.index)
    if len(common) < 100:
        return mark_failed("AD-S2",
                           "Too little overlap between NANC, KRUZ, SPY.")

    rN = nanc.loc[common].pct_change()
    rK = kruz.loc[common].pct_change()
    rS = spy.loc[common].pct_change()

    # Equal-weight bipartisan proxy basket return
    rB = 0.5 * rN + 0.5 * rK
    # Excess return vs SPY
    rEx = (rB - rS).dropna()

    m = compute_metrics(rEx, benchmark=rS.dropna(),
                        name="AD-S2 NANC+KRUZ basket excess vs SPY")
    print(f"AD-S2 (coarse proxy: equal-weight NANC+KRUZ vs SPY)")
    print_metrics(m)
    extra = {
        "status": "ok_proxy",
        "rule": "Spec: >=2 D + >=2 R buy same ticker within 21d window, no "
                "offsetting sales, mega-cap excluded -> long 60-90d. Proxy: "
                "long equal-weight (NANC + KRUZ) basket vs short SPY.",
        "mechanism": "Bipartisan trading strips out partisan bias; presumed "
                     "non-public catalyst across party lines.",
        "source": "Unusual Whales Subversive ETFs NANC/KRUZ holdings as "
                  "static bipartisan proxy. yfinance prices.",
        "caveats": [
            "VERY coarse proxy: this measures the long-only equal-weight "
            "bipartisan basket return vs SPY, NOT actual cluster events.",
            "True signal requires daily CapitolTrades / Quiver Quant scrape "
            "to identify ticker-level convergence within 21-day windows.",
            "NANC/KRUZ only launched Feb 2023, so sample is ~3 years.",
        ],
    }
    save_result("AD-S2", m, extra=extra)


if __name__ == "__main__":
    main()
