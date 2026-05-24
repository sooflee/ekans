"""
A02 Turn-of-Month
Long SPY from close of T-1 (last trading day of month) through close of T+3
(3rd trading day of new month). Cash otherwise.
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

    # For each month, find the last trading day (T-1) and first 3 trading days of next month
    # Build a boolean mask: in-window means position=1
    df = pd.DataFrame({"ret": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Position at close of date d => earns next day's return.
    # We want to be "in" from close of T-1 through close of T+3.
    # Implementation: pos[d] = 1 if d is one of {last trading day of month,
    # first 3 trading days of next month}. Then long_short_pnl shifts by 1
    # and applies to next-day return -> we earn returns on days T, T+1, T+2, T+3.
    last_of_month = df.groupby(["year", "month"]).apply(lambda g: g.index[-1])
    last_set = set(last_of_month.values)

    # First 3 trading days of each month
    first_n = df.groupby(["year", "month"]).apply(lambda g: list(g.index[:3]))
    first_set = set()
    for arr in first_n:
        for d in arr:
            first_set.add(d)

    # Position date d: if d is last-of-month OR if d is one of the first 2 days of a month
    # (because pos shifted by 1 day produces returns on T, T+1, T+2, T+3 — that's
    # the 1st, 2nd, 3rd trading day of new month, and we still want to capture the
    # T+3 day's return so we need pos at T+2 too. Last-of-month pos earns T return.)
    # First 3 trading days are T, T+1, T+2. To earn T+3 return, we need pos at T+2.
    # So the position-holding dates = {last-of-month} ∪ {1st, 2nd, 3rd trading day of month}
    pos = pd.Series(0.0, index=idx)
    pos[pos.index.isin(last_set)] = 1.0
    pos[pos.index.isin(first_set)] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A02 Turn-of-Month")
    print_metrics(m)
    save_result("A02_turn_of_month", m, extra={
        "status": "ok",
        "rule": "Long SPY from close of last trading day of month through close of T+3 of next month.",
        "universe": "SPY",
        "source": "Ariel 1987; Lakonishok-Smidt 1988",
    })


if __name__ == "__main__":
    main()
