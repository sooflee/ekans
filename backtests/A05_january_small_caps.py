"""
A05 January Effect on Small Caps
Long IWM in January only; cash else. Compare IWM excess over SPY (long IWM / short SPY in Jan).
We report two flavors:
  - long-only IWM in January (vs SPY benchmark)
  - long IWM / short SPY in January (zero-cost)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["IWM", "SPY"], start="2000-01-01")
    rets = daily_returns(px)
    rets = rets.dropna()

    idx = rets.index
    months = idx.month

    # Long-only IWM in January, cash else.
    pos_iwm = pd.Series(0.0, index=idx)
    pos_iwm[months == 1] = 1.0
    pnl_long = pos_iwm.shift(1) * rets["IWM"]

    # Long IWM / short SPY in January only (zero-cost spread)
    pnl_spread = pos_iwm.shift(1) * (rets["IWM"] - rets["SPY"])

    m_long = compute_metrics(pnl_long.dropna(), benchmark=rets["SPY"], name="A05 IWM-Jan long-only")
    m_spread = compute_metrics(pnl_spread.dropna(), benchmark=rets["SPY"], name="A05 IWM-SPY Jan spread")

    print_metrics(m_long)
    print_metrics(m_spread)

    # Save primary signal as the IWM-SPY spread (the canonical small-cap Jan effect)
    save_result("A05_january_small_caps", m_spread, extra={
        "status": "ok",
        "rule": "Long IWM / short SPY in January only; flat otherwise.",
        "universe": "IWM vs SPY",
        "source": "Banz 1981; Keim 1983",
        "long_only_variant": m_long,
    })


if __name__ == "__main__":
    main()
