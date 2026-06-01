"""PL663 — CMS IRA Negotiation List Annual Selection - Short CVS/CI Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL663_cms_ira_list_short_cvs_ci"

    # Known CMS IRA negotiation list publication dates
    known_events = [
        "2024-02-01",
        "2025-01-17",
    ]

    try:
        px = load_prices(["CVS", "CI", "TEVA", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "no price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build PnL: short CVS + CI (equal weight), long 50% TEVA hedge
    # Position: -0.5 CVS, -0.5 CI, +0.5 TEVA -> net = short pharma PBM
    pnl = pd.Series(0.0, index=ret.index)
    events = []

    hold = 40  # trading days

    for ed in known_events:
        event_date = pd.Timestamp(ed)
        # find first trading day >= event_date
        mask = ret.index >= event_date
        if mask.sum() == 0:
            continue
        start_idx = ret.index[mask][0]
        p = ret.index.get_loc(start_idx)
        end_idx = min(p + hold, len(ret))

        window_ret = ret.iloc[p:end_idx]

        # Short CVS + CI equally weighted, long 0.5 TEVA
        if "CVS" not in window_ret.columns or "CI" not in window_ret.columns:
            continue
        if "TEVA" not in window_ret.columns:
            teva_ret = pd.Series(0.0, index=window_ret.index)
        else:
            teva_ret = window_ret["TEVA"]

        # -0.5*CVS - 0.5*CI + 0.5*TEVA daily pnl
        event_pnl = (
            -0.5 * window_ret["CVS"].fillna(0)
            - 0.5 * window_ret["CI"].fillna(0)
            + 0.5 * teva_ret.fillna(0)
        )

        # Avoid double-counting overlapping events
        already_in = (pnl.iloc[p:end_idx] != 0).any()
        if not already_in:
            pnl.iloc[p:end_idx] = event_pnl.values

        # Compute event-level metrics for diagnostics
        total_return = float((1 + event_pnl).prod() - 1)
        spy_window = spy_r.iloc[p:end_idx]
        spy_total = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else None

        events.append({
            "event_date": ed,
            "start": str(start_idx.date()),
            "pnl_return": round(total_return, 4),
            "spy_return": round(spy_total, 4) if spy_total is not None else None,
        })

    print(f"Events processed: {len(events)}")
    for ev in events:
        print(f"  {ev}")

    # Filter to active (non-zero) days
    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="CMS IRA Neg List Short CVS/CI")
    save_result(sid, m, extra={
        "rule": "Short CVS+CI equal-weight, long 0.5 TEVA for 40 days following CMS IRA negotiation list publication (~Feb 1 annually)",
        "mechanism": "CMS Medicare drug price negotiation list triggers PBM/insurer margin pressure, benefits generic manufacturers",
        "source": "CMS IRA negotiation announcement dates; yfinance",
        "n_events": len(events),
        "events": events,
        "status": "ok",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
