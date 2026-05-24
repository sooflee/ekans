"""
P10 GRNY ETF (Granny Shots) total-return vs SPY since inception.

Note: GRNY launched Nov 7 2024 on yfinance data (spec said Nov 2023, but
actual inception per yfinance is Nov 2024). We use whatever data exists.

PnL = GRNY daily return - SPY daily return (relative outperformance).
We also report a "long GRNY" absolute strategy for completeness.
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
    px = load_prices(["GRNY", "SPY"], start="2023-01-01").dropna()
    rets = px.pct_change().dropna()
    # Relative-strategy PnL: long GRNY, short SPY (equal-weight pair).
    rel = rets["GRNY"] - rets["SPY"]
    m_rel = compute_metrics(rel, benchmark=rets["SPY"], name="P10 GRNY - SPY (relative)")
    print_metrics(m_rel)

    # Also: simple long GRNY total return.
    m_abs = compute_metrics(rets["GRNY"], benchmark=rets["SPY"], name="P10 GRNY long (absolute)")
    print_metrics(m_abs)

    save_result("P10_granny_shots_etf", m_rel, extra={
        "status": "ok",
        "rule": "Long GRNY vs short SPY since GRNY inception (pair PnL).",
        "mechanism": "Granny Shots: concentrated 'forever stocks' theme of intersecting themes (Tom Lee). Track total-return outperformance.",
        "universe": "GRNY long / SPY short pair.",
        "grny_start": str(px.index[0].date()),
        "n_days": int(len(rel)),
        "absolute_grny_metrics": m_abs,
        "source": "Tom Lee / FundStrat 'Granny Shots' (Phase 1P, YouTube)",
    })


if __name__ == "__main__":
    main()
