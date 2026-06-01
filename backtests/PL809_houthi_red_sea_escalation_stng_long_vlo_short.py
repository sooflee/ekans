"""PL809_houthi_red_sea_escalation_stng_long_vlo_short
Houthi Red Sea Drone-Cluster Escalation -> Long STNG (Product Tanker) / Short VLO (Refiner)

Event-study pair trade: uses known Houthi Red Sea escalation events plus
a rolling STNG momentum signal to identify product tanker rate spikes.
Long STNG / short VLO at 0.8x notional, hold 30 calendar days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL809_houthi_red_sea_escalation_stng_long_vlo_short"
    tickers = ["STNG", "VLO", "SPY"]

    try:
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    stng_r = ret["STNG"].fillna(0)
    vlo_r = ret["VLO"].fillna(0)
    stng_px = px["STNG"]

    # Primary signal: STNG 10-day return > 15% (momentum proxy for tanker rate spike)
    # This proxies the tanker rate escalation signal when event news is not available
    stng_10d_ret = stng_px.pct_change(10)
    stng_60d_avg = stng_px.rolling(60).mean()

    # Align all series to ret.index to avoid length mismatches
    stng_10d_ret = stng_10d_ret.reindex(ret.index)
    stng_60d_avg = stng_60d_avg.reindex(ret.index)
    stng_px_aligned = stng_px.reindex(ret.index)

    # Precondition: STNG not >20% above its 60d average (avoid chasing)
    not_chasing = (stng_px_aligned / stng_60d_avg - 1) <= 0.20

    prev_10d = stng_10d_ret.shift(1)
    # Cross from below 15%
    momentum_signal = (prev_10d <= 0.15) & (stng_10d_ret > 0.15) & not_chasing

    # Also explicitly seed known Houthi escalation events
    known_events = ["2023-11-19", "2024-01-09"]

    signal_dates = list(ret.index[momentum_signal.fillna(False)])
    for ev in known_events:
        dt = pd.Timestamp(ev)
        # Add the known event dates if not already captured by momentum signal
        future = ret.index[ret.index >= dt]
        if len(future) > 0:
            signal_dates.append(future[0])

    signal_dates = sorted(set(signal_dates))

    hold_calendar_days = 30
    min_gap_days = 30
    long_notional = 1.0
    short_notional = 0.8

    positions_long = pd.Series(0.0, index=ret.index)
    positions_short = pd.Series(0.0, index=ret.index)
    last_exit_date = pd.Timestamp("2000-01-01")

    for sig_dt in signal_dates:
        if (sig_dt - last_exit_date).days < min_gap_days:
            continue

        future = ret.index[ret.index > sig_dt]
        if len(future) == 0:
            continue
        entry_date = future[0]

        hard_exit_dt = entry_date + pd.Timedelta(days=hold_calendar_days)
        hard_exit_candidates = ret.index[ret.index >= hard_exit_dt]
        if len(hard_exit_candidates) == 0:
            exit_date = ret.index[-1]
        else:
            exit_date = hard_exit_candidates[0]

        hold_mask = (ret.index >= entry_date) & (ret.index <= exit_date)
        positions_long.loc[ret.index[hold_mask]] = long_notional
        positions_short.loc[ret.index[hold_mask]] = -short_notional
        last_exit_date = exit_date

    pnl = (positions_long.shift(1).fillna(0) * stng_r +
           positions_short.shift(1).fillna(0) * vlo_r)
    pnl = pnl.reindex(spy_r.index).fillna(0)

    m = compute_metrics(pnl, benchmark=spy_r, name="Houthi Red Sea STNG Long VLO Short")
    save_result(sid, m, extra={
        "rule": "Long STNG (1x) / Short VLO (0.8x) when STNG 10d momentum >15% (proxy for tanker rate spike / Red Sea escalation). Hold 30 calendar days. Min 30d between signals. Precondition: STNG not >20% above 60d avg.",
        "mechanism": "Houthi Red Sea attacks force product tankers (STNG) to reroute Cape of Good Hope, spiking TC2/TC7 rates and STNG earnings. VLO (Gulf Coast refiner) faces higher import freight costs. Pair captures the tanker-vs-refiner spread.",
        "source": "Known events: 2023-11-19 Galaxy Leader seizure, 2024-01-09 US-UK retaliatory strikes. STNG 10d momentum >15% used as supplementary signal proxy.",
    })


if __name__ == "__main__":
    main()
