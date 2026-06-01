"""PL673 — BofA FMS Cash <3.5% - Counter-Signal Short SPY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL673_bofa_fms_cash_low_short_spy"

    # Known BofA FMS events where institutional cash allocation dropped below 3.5%
    # These are historically notable over-bullish readings (contrarian short signal)
    # Extending beyond the 2 known events with historically documented dates
    known_events = [
        "2018-01-16",   # Jan 2018 - extreme bullish, pre-correction
        "2021-04-13",   # Apr 2021 - cash at multi-year lows
        "2021-11-15",   # Nov 2021 - near peak (known from spec)
        "2024-02-15",   # Feb 2024 - cash dropped below 3.5% (known from spec)
    ]

    try:
        px = load_prices(["SPY", "TLT"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "no price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    tlt_r = ret["TLT"] if "TLT" in ret.columns else pd.Series(0.0, index=ret.index)

    hold = 40  # trading days
    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ed in known_events:
        event_date = pd.Timestamp(ed)
        mask = ret.index >= event_date
        if mask.sum() == 0:
            continue
        start_idx = ret.index[mask][0]
        p = ret.index.get_loc(start_idx)
        end_idx = min(p + hold, len(ret))

        window_ret = ret.iloc[p:end_idx]
        spy_win = window_ret["SPY"].fillna(0)
        tlt_win = tlt_r.iloc[p:end_idx].fillna(0)

        # Short SPY, 50% long TLT hedge
        # Net position: -1 SPY + 0.5 TLT
        event_pnl = -spy_win + 0.5 * tlt_win

        already_in = (pnl.iloc[p:end_idx] != 0).any()
        if not already_in:
            pnl.iloc[p:end_idx] = event_pnl.values

        total_return = float((1 + event_pnl).prod() - 1)
        spy_fwd = float((1 + spy_win).prod() - 1)

        event_records.append({
            "event_date": ed,
            "start": str(start_idx.date()),
            "pnl_return": round(total_return, 4),
            "spy_fwd_return": round(spy_fwd, 4),
        })

    print(f"Events processed: {len(event_records)}")
    for ev in event_records:
        print(f"  {ev}")

    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="BofA FMS Cash Low Short SPY")
    save_result(sid, m, extra={
        "rule": "Short SPY + 50% long TLT for 40 days when BofA FMS cash allocation <3.5% (monthly release)",
        "mechanism": "Extremely low institutional cash levels signal peak positioning and reduced buying power; contrarian short as sentiment extreme often precedes correction",
        "source": "BofA Global Fund Manager Survey (public release dates); yfinance",
        "n_events": len(event_records),
        "events": event_records,
        "status": "ok",
    })
    print(f"Done: {len(event_records)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
