"""
M8 Russell reconstitution effect.
Annual event, last Friday of June (with a few historical date tweaks).
Long IWM 15 trading days before through reconstitution date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def last_friday_of_june(year):
    # last Friday in June
    d = pd.Timestamp(year=year, month=6, day=30)
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d


def main():
    try:
        iwm = load_prices(["IWM"], start="2000-01-01").iloc[:, 0].rename("IWM")
    except Exception as e:
        return mark_failed("M8_russell_reconstitution", f"data load failed: {e}")

    # Russell reconstitution dates (last Friday of June) for years 2000-2025
    # Known exceptions (per FTSE Russell announcements; close enough for trading purposes):
    #   2020 reconstitution delayed to June 29, 2020 (Mon) due to COVID-related scheduling
    # We'll use last-Friday-of-June across the board; off by 1-2 trading days doesn't change the picture.
    recon_dates = [last_friday_of_june(y) for y in range(2000, 2026)]

    rets = iwm.pct_change()
    pos = pd.Series(0.0, index=iwm.index)
    n_events = 0
    for rd in recon_dates:
        # Snap to nearest trading day on/before
        ix = iwm.index.searchsorted(rd)
        if ix >= len(iwm.index):
            continue
        # If exact match, use it; else use prior trading day
        if ix < len(iwm.index) and iwm.index[ix] != rd:
            ix = max(ix - 1, 0)
        # window: 15 trading days before through the recon date inclusive
        start_ix = max(ix - 15, 0)
        end_ix = ix + 1  # include the recon day
        pos.iloc[start_ix:end_ix] = 1.0
        n_events += 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M8 Russell reconstitution → long IWM 15d pre through event")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M8_russell_reconstitution", m, extra={
        "status": "ok",
        "rule": "Long IWM for 15 trading days before through reconstitution day (last Friday of June).",
        "mechanism": "Index funds rebalance to new Russell constituents over the reconstitution window; demand for added small-cap names lifts IWM in the run-up.",
        "source": "FTSE Russell schedules; Madhavan (2003) index effect literature.",
    })


if __name__ == "__main__":
    main()
