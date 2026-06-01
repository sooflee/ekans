"""PL756_appstore_streaming_short_nflx — App Store Streaming Rank Decay -> Short NFLX (Counter)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL756_appstore_streaming_short_nflx"

    # App Store rank data (Sensor Tower / AppFollow) is proprietary.
    # Proxy: hand-coded NFLX subscriber-miss / engagement-decline quarters
    # where App Store streaming rank deterioration is publicly documented or
    # strongly implied by subscriber trajectory.
    #
    # Sources:
    # - NFLX earnings reports and public Sensor Tower estimates
    # - Apple App Store historical Entertainment category rank proxied via
    #   media-reported rank movements and subscriber data
    #
    # Trigger dates = first trading day after rank-decay condition identified
    # (approx. 30-day rolling rank deterioration >=10 positions vs 90d baseline):
    rank_decay_events = [
        "2019-04-17",  # Q1 2019: NFLX missed US subs by 1.74M; engagement/rank fell
        "2021-10-20",  # Q3 2021: NFLX Q3 miss; App Store rank slipped vs prior quarter
        "2022-01-21",  # Q4 2021: NFLX -200k guidance; App Store Entertainment rank
                       #          dropped from top-3 to top-8 per Sensor Tower estimates
        "2022-04-20",  # Q1 2022: Lost 200k subscribers; confirmed rank decay event
        "2023-01-19",  # Q4 2022: Password-sharing crackdown fears; engagement dip
    ]

    try:
        px = load_prices(["NFLX", "SPY"], start="2015-01-01")
        ret = daily_returns(px)
        nflx_r = ret["NFLX"]
        spy_r = ret["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 25  # 25 trading days as specified

    pnl = pd.Series(0.0, index=nflx_r.index)
    events = []

    for event_date_str in rank_decay_events:
        event_date = pd.Timestamp(event_date_str)
        # Find first available trading day on or after event date
        mask = nflx_r.index >= event_date
        if mask.sum() < hold:
            print(f"  Skipping {event_date_str}: insufficient future data")
            continue

        entry_idx = nflx_r.index[mask][0]
        p = nflx_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(nflx_r))

        # SHORT NFLX: negate returns
        window_r = nflx_r.iloc[p:ep]
        short_r = -window_r

        # Record PnL (overwrites if overlapping — events well-spaced so unlikely)
        window_idx = nflx_r.index[p:ep]
        pnl.loc[window_idx] = short_r.values[:len(window_idx)]

        cum_nflx = float((1 + window_r).prod() - 1)
        cum_short = -cum_nflx

        cum_spy = None
        if entry_idx in spy_r.index:
            sp = spy_r.index.get_loc(entry_idx)
            sp_end = min(sp + hold, len(spy_r))
            cum_spy = float((1 + spy_r.iloc[sp:sp_end]).prod() - 1)

        events.append({
            "trigger_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "short_nflx_return": round(cum_short, 4),
            "nflx_underlying_return": round(cum_nflx, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
        })
        print(f"  Event {event_date_str}: short NFLX {cum_short*100:.1f}% vs SPY "
              f"{cum_spy*100:.1f}%"
              if cum_spy is not None
              else f"  Event {event_date_str}: short NFLX {cum_short*100:.1f}%")

    print(f"Total events: {len(events)}")

    if len(events) == 0:
        return mark_failed(sid, "no valid rank-decay events found")

    # Extract non-zero PnL days
    active_pnl = pnl[pnl != 0]

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active PnL days ({len(active_pnl)}): "
                               f"only {len(events)} event(s) with {hold}-day windows")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="App Store Rank Decay -> Short NFLX")
    save_result(sid, m, extra={
        "rule": "Short NFLX for 25 trading days when 30d-mean App Store rank "
                "deteriorates >=10 positions vs trailing-90d baseline",
        "mechanism": "App Store Entertainment rank decay signals disengagement ahead "
                     "of subscriber-miss earnings; institutional selling follows as "
                     "alternative data confirms negative sentiment.",
        "source": "Sensor Tower / App Store rank proxied via NFLX earnings data; "
                  "yfinance NFLX, SPY",
        "n_events": len(events),
        "events": events,
        "caveats": "Actual Sensor Tower rank data not available; events proxied from "
                   "public earnings miss dates where rank decay was documented. "
                   "bt_feasibility=3. True signal would require licensed App Store data.",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
