"""
A13 Presidential cycle year 3.
Long SPY only during the 3rd calendar year of each presidential term; cash else.
Year-3 years: 1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023.
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
    idx = rets.index

    year3 = {2003, 2007, 2011, 2015, 2019, 2023}
    pos = pd.Series(0.0, index=idx)
    pos[idx.year.isin(year3)] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A13 Presidential Y3")
    print_metrics(m)
    save_result("A13_presidential_year3", m, extra={
        "status": "ok",
        "rule": "Long SPY only during year-3 of each US presidential term (2003, 2007, 2011, 2015, 2019, 2023). Cash else.",
        "universe": "SPY",
        "source": "Stock Trader's Almanac; Herbst-Slinkman 1984",
        "year3_years_in_sample": sorted(year3),
    })


if __name__ == "__main__":
    main()
