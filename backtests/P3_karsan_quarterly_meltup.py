"""
P3 Karsan Quarterly Meltup.

Quarterly OPEX = 3rd Friday of Mar / Jun / Sep / Dec.
Long SPY 10 trading days before quarterly OPEX; exit at quarterly OPEX open.
We approximate "OPEX open" by closing the position at the close of the
trading day immediately preceding OPEX-Friday (i.e., the prior day's close,
which is the closest daily-bar surrogate for Friday-open).
"""
import sys
import calendar
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def third_friday(year, month):
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2] if len(fridays) >= 3 else None


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    n_windows = 0
    for y in range(idx[0].year, idx[-1].year + 1):
        for m in (3, 6, 9, 12):
            tf = third_friday(y, m)
            if tf is None:
                continue
            opex = pd.Timestamp(tf)
            i_opex = idx.searchsorted(opex, side="right") - 1
            if i_opex < 11:
                continue
            i_entry = i_opex - 10           # close 10 trading days before OPEX
            i_exit = i_opex - 1             # exit at OPEX-1 close ~= OPEX-open
            if i_entry < 0 or i_exit <= i_entry:
                continue
            hold_dates = idx[i_entry:i_exit]
            pos.loc[hold_dates] = 1.0
            n_windows += 1

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="P3 Karsan Quarterly Meltup")
    print_metrics(m)
    save_result("P3_karsan_quarterly_meltup", m, extra={
        "status": "ok",
        "rule": "Long SPY 10 trading days before quarterly OPEX (Mar/Jun/Sep/Dec 3rd Friday); exit at OPEX open (proxied by OPEX-1 close).",
        "mechanism": "Karsan: dealer-positioning meltup into quarterly opex as charm flows push spot up.",
        "universe": "SPY",
        "n_windows": n_windows,
        "source": "Cem Karsan (YouTube/Twitter, Phase 1P)",
    })


if __name__ == "__main__":
    main()
