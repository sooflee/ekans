"""
A06 FOMC even-week effect (Cieslak-Morse-Vissing-Jorgensen 2019).
Long SPY only during "even" weeks (0, 2, 4, 6) of the FOMC cycle (where week 0 is the
week containing the FOMC meeting decision).
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

    fomc = fomc_dates_ts()
    fomc = [d for d in fomc if (idx[0] <= d <= idx[-1] + pd.Timedelta(days=60))]

    # For each trading day, compute days since most recent FOMC meeting
    # and bucket into weeks. We use calendar days // 7 to get week index.
    fomc_arr = pd.Series(sorted(fomc))
    # For each idx date, find the index of the most recent FOMC date <= that date
    # (i.e., reference to last meeting). Use searchsorted.
    sorted_fomc = pd.DatetimeIndex(sorted(fomc))
    pos_in_fomc = sorted_fomc.searchsorted(idx, side="right") - 1
    # days since last FOMC
    days_since = np.where(
        pos_in_fomc >= 0,
        (idx - sorted_fomc[np.clip(pos_in_fomc, 0, len(sorted_fomc) - 1)]).days,
        np.nan,
    )
    week_idx = np.where(np.isnan(days_since), np.nan, np.floor(np.asarray(days_since, dtype=float) / 7.0))

    pos = pd.Series(0.0, index=idx)
    # Even weeks: 0, 2, 4, 6
    even_mask = np.isin(week_idx, [0, 2, 4, 6])
    pos[even_mask] = 1.0

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A06 FOMC even-week")
    print_metrics(m)
    save_result("A06_fomc_even_week", m, extra={
        "status": "ok",
        "rule": "Long SPY only in weeks 0,2,4,6 after each FOMC meeting (calendar-day // 7 from most recent meeting).",
        "universe": "SPY",
        "source": "Cieslak, Morse, Vissing-Jorgensen (JF 2019)",
        "n_fomc_events": len(fomc),
    })


if __name__ == "__main__":
    main()
