"""PL718_cpi_cordcut_long_spot — CPI Cable <-2% + Streaming >+5% -> Long SPOT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL718_cpi_cordcut_long_spot"
    # SPOT IPO April 2018
    start = "2018-04-01"

    try:
        px = load_prices(["SPOT", "SPY"], start=start)
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        # CUSR0000SERA02 = Cable & satellite television service CPI
        cable_cpi = load_fred("CUSR0000SERA02", start="2017-01-01")
        # CUSR0000SEHF02 = Video streaming and subscription services
        stream_cpi = load_fred("CUSR0000SEHF02", start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load FRED: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["SPOT", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    spot_r = ret["SPOT"]

    # Compute YoY changes for each series
    cable_yoy = cable_cpi["CUSR0000SERA02"].pct_change(12)
    stream_yoy = stream_cpi["CUSR0000SEHF02"].pct_change(12)

    # Find months where cable YoY < -2% AND streaming YoY > +5%
    # CPI monthly releases are typically 2 weeks into following month
    # Use the end of the reported month as approximate release signal
    cable_yoy.index = pd.to_datetime(cable_yoy.index)
    stream_yoy.index = pd.to_datetime(stream_yoy.index)

    # Align to same monthly index
    cpi_df = pd.DataFrame({"cable_yoy": cable_yoy, "stream_yoy": stream_yoy}).dropna()
    signal_months = cpi_df[
        (cpi_df["cable_yoy"] < -0.02) & (cpi_df["stream_yoy"] > 0.05)
    ]

    all_dates = ret.index
    pnl = pd.Series(0.0, index=all_dates)
    n_events = 0
    hold_days = 45

    for month_start, row in signal_months.iterrows():
        # BLS releases data ~2 weeks into the following month
        # Use 15th of the following month as approximate release date
        release_approx = month_start + pd.DateOffset(months=1, days=14)
        valid = all_dates[all_dates >= release_approx]
        if len(valid) == 0:
            continue
        # Skip if before SPOT IPO
        if valid[0] < pd.Timestamp("2018-05-01"):
            continue
        entry_date = valid[0]
        idx_start = all_dates.get_loc(entry_date)
        idx_end = min(idx_start + hold_days, len(all_dates))
        window = all_dates[idx_start:idx_end]
        pnl.loc[window] += spot_r.reindex(window).fillna(0)
        n_events += 1

    if n_events == 0:
        return mark_failed(sid, "no qualifying CPI signal months found")

    m = compute_metrics(pnl, benchmark=spy_r, name="CPI Cord-Cut -> Long SPOT")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "BLS CPI monthly: cable & satellite service YoY < -2% AND streaming service YoY > +5%; long SPOT for 45 trading days after release",
        "mechanism": "Accelerating cord-cutting with rising streaming consumption validates Spotify-adjacent streaming growth thesis; signals consumer shift to on-demand audio/video accelerates SPOT subscriber growth",
        "source": "FRED CUSR0000SERA02 (cable CPI), CUSR0000SEHF02 (video streaming CPI)",
        "signal_months": [str(m) for m in signal_months.index.tolist()],
    })


if __name__ == "__main__":
    main()
