"""
A04 First Five Days of January
If SPY return Jan trading days 1-5 > 0, long SPY rest of year; else cash.
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

    df = pd.DataFrame({"ret": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month

    pos = pd.Series(0.0, index=idx)
    years = sorted(df["year"].unique())
    for y in years:
        jan = df[(df["year"] == y) & (df["month"] == 1)]
        if len(jan) < 5:
            continue
        # First 5 trading days = jan.index[0..4]. Their returns (rets at those indices)
        # is the indicator. Computed using realized close-to-close returns of those days.
        first5_rets = rets.loc[jan.index[:5]]
        signal = first5_rets.sum() > 0
        if signal:
            # Hold from day after 5th trading day through Dec 31. Pos at close of day 5
            # earns return on day 6 onward.
            start_pos_date = jan.index[4]  # position from end of day 5
            year_end_dates = df[df["year"] == y].index
            hold_dates = year_end_dates[year_end_dates >= start_pos_date]
            pos.loc[hold_dates] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A04 First Five Days")
    print_metrics(m)
    save_result("A04_first_five_days", m, extra={
        "status": "ok",
        "rule": "If SPY return over Jan trading days 1-5 > 0, long SPY for remainder of year; else cash. Computed on year-by-year basis.",
        "universe": "SPY",
        "source": "Hirsch (Stock Trader's Almanac)",
    })


if __name__ == "__main__":
    main()
