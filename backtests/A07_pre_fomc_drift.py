"""
A07 Pre-FOMC drift (Lucca-Moench 2015).
Long SPY from close of T-1 (day before scheduled FOMC) to close of T (FOMC day).
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
from _fomc_dates import fomc_dates_ts


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    fomc = sorted([d for d in fomc_dates_ts() if idx[0] <= d <= idx[-1]])

    pos = pd.Series(0.0, index=idx)
    # For each FOMC date, find the trading day at or before that date (= T),
    # and the trading day immediately before that (= T-1).
    # We want to be holding from close of T-1 through close of T.
    # That means pos at T-1 (so we earn return on T).
    # Long T-1 close to T close = return on day T.
    held = 0
    for d in fomc:
        # Find T (FOMC trading day) -> first idx <= d (FOMC days are usually trading days)
        pos_in_idx = idx.searchsorted(d, side="right") - 1
        if pos_in_idx <= 0:
            continue
        t_day = idx[pos_in_idx]
        if t_day != d:
            # If d itself isn't a trading day, this is unusual; FOMC meetings are weekdays
            # but check anyway. Treat the prior trading day as T-1 -> hold its close.
            continue
        t_minus_1 = idx[pos_in_idx - 1]
        pos.loc[t_minus_1] = 1.0
        held += 1

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A07 Pre-FOMC drift")
    print_metrics(m)
    save_result("A07_pre_fomc_drift", m, extra={
        "status": "ok",
        "rule": "Long SPY from close of T-1 to close of T (FOMC decision day). Pos=1 at T-1 close, earns return on day T.",
        "universe": "SPY",
        "source": "Lucca & Moench (JF 2015)",
        "n_events_held": held,
    })


if __name__ == "__main__":
    main()
