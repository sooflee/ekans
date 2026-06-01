"""PL753_hcm_add_short_lrn — Title IV HCM Add -> Short LRN (Counter)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL753_hcm_add_short_lrn"

    # Hand-coded DOE HCM (Heightened Cash Monitoring) events for LRN (Stride/K12)
    # Source: DOE quarterly HCM list disclosures
    # K12 Inc (renamed Stride Inc / LRN in 2020) has been subject to DOE scrutiny.
    # HCM1 placements for LRN:
    #   - 2013-Q2: K12 Inc placed on HCM list amid DOE scrutiny of virtual schools
    #   - 2022-Q2: Stride Inc (LRN) added to HCM1 list (confirmed per strategy notes 2022-Q3)
    # Note: DOE publishes quarterly, so we use approximate disclosure dates.
    # HCM list publication dates (announcement/media pickup date used as trigger):
    hcm_events = [
        "2013-07-01",   # K12 Inc / HCM placement circa Q2 2013
        "2022-07-05",   # Stride (LRN) HCM1 placement disclosed Q2 2022 (Q3 media coverage)
    ]

    try:
        px = load_prices(["LRN", "SPY"], start="2010-01-01")
        ret = daily_returns(px)
        lrn_r = ret["LRN"]
        spy_r = ret["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 30  # 30 trading days as specified

    pnl = pd.Series(0.0, index=lrn_r.index)
    events = []

    for event_date_str in hcm_events:
        event_date = pd.Timestamp(event_date_str)
        # Find first available trading day on or after event date
        mask = lrn_r.index >= event_date
        if mask.sum() < hold:
            print(f"  Skipping {event_date_str}: insufficient future data")
            continue

        entry_idx = lrn_r.index[mask][0]
        p = lrn_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(lrn_r))

        # SHORT LRN: negate returns
        window_r = lrn_r.iloc[p:ep]
        short_r = -window_r

        # Avoid double-counting overlapping windows
        window_idx = lrn_r.index[p:ep]
        pnl.loc[window_idx] = short_r.values[:len(window_idx)]

        cum_lrn = float((1 + window_r).prod() - 1)
        cum_short = -cum_lrn

        spy_p = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
        cum_spy = None
        if spy_p is not None:
            sp_end = min(spy_p + hold, len(spy_r))
            cum_spy = float((1 + spy_r.iloc[spy_p:sp_end]).prod() - 1)

        events.append({
            "trigger_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "short_lrn_return": round(cum_short, 4),
            "lrn_underlying_return": round(cum_lrn, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
        })
        print(f"  Event {event_date_str}: short LRN {cum_short*100:.1f}% vs SPY {cum_spy*100:.1f}%"
              if cum_spy is not None else f"  Event {event_date_str}: short LRN {cum_short*100:.1f}%")

    print(f"Total events: {len(events)}")

    if len(events) == 0:
        return mark_failed(sid, "no valid HCM events found")

    # Extract non-zero PnL days
    active_pnl = pnl[pnl != 0]

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active PnL days ({len(active_pnl)}): "
                               f"only {len(events)} HCM event(s) found for LRN, "
                               f"each covering {hold} days = {len(events)*hold} days total")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="HCM Add -> Short LRN")
    save_result(sid, m, extra={
        "rule": "Short LRN for 30 trading days when DOE adds operator to HCM1/HCM2 list",
        "mechanism": "DOE HCM placement signals enrollment/cash-flow risk; "
                     "institutional operators face Title IV funding review, "
                     "forcing enrollment freezes and regulatory overhang.",
        "source": "DOE Title IV HCM quarterly disclosures; yfinance LRN, SPY",
        "n_events": len(events),
        "events": events,
        "caveats": "Very sparse event history for LRN specifically; "
                   "K12/Stride has appeared on HCM list infrequently. "
                   "bt_feasibility=3 reflects limited event count.",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
