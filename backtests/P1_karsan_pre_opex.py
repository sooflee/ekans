"""
P1 Karsan Pre-OPEX (Phase 1P, YouTube trader content).

Rule (per Karsan): Long SPY from Monday close preceding OPEX week through
Wednesday open of OPEX week. We approximate as: hold from close of T-7
(prior-week's Friday-trading-day proxy) through close of T-2 of OPEX week
(Wednesday close of OPEX week).

Operationally: enter at close on the trading day that falls on (or just before)
the Monday before OPEX-Friday, exit at close on the trading day that falls on
(or just before) the Wednesday of OPEX-Friday's week.
"""
import sys
import calendar
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
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
    years = range(idx[0].year, idx[-1].year + 1)
    n_windows = 0
    for y in years:
        for m in range(1, 13):
            tf = third_friday(y, m)
            if tf is None:
                continue
            opex = pd.Timestamp(tf)
            # Monday of OPEX week = OPEX-Friday minus 4 days
            opex_monday = opex - pd.Timedelta(days=4)
            # Wednesday of OPEX week = OPEX-Friday minus 2 days
            opex_wed = opex - pd.Timedelta(days=2)
            # Spec: enter at Monday close (closest trading day at/before Mon),
            # exit at Wed open (~ approx Wed close, but use Wed close for daily-bar PnL)
            i_mon = idx.searchsorted(opex_monday, side="right") - 1
            i_wed = idx.searchsorted(opex_wed, side="right") - 1
            if i_mon < 0 or i_wed < 0 or i_wed <= i_mon:
                continue
            # Pos at i_mon..(i_wed-1) earns returns on (i_mon+1)..i_wed
            hold_dates = idx[i_mon:i_wed]
            pos.loc[hold_dates] = 1.0
            n_windows += 1

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="P1 Karsan Pre-OPEX")
    print_metrics(m)
    save_result("P1_karsan_pre_opex", m, extra={
        "status": "ok",
        "rule": "Long SPY from Monday-close before OPEX through Wednesday-close of OPEX week (~3 trading days).",
        "mechanism": "Karsan: dealer gamma positioning and pre-OPEX vol compression drift.",
        "universe": "SPY",
        "n_windows": n_windows,
        "source": "Cem Karsan (YouTube/Twitter, Phase 1P)",
    })


if __name__ == "__main__":
    main()
