"""PL665 — HRSA 340B Child-Site Net Adds Surge - Long CVS Short DOC"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL665_hrsa_340b_long_cvs_short_doc"

    # Known HRSA OPAIS events where 340B child-site registrations surged
    # (>1.5 stdev above 8q trailing mean). Using known events from strategy spec.
    known_events = [
        "2023-09-30",
        "2024-12-31",
    ]

    try:
        px = load_prices(["CVS", "DOC", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "no price data")

    # DOC started trading ~2023 after merger
    if "DOC" not in px.columns or px["DOC"].dropna().empty:
        return mark_failed(sid, "DOC ticker not available or no data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    hold = 50  # trading days
    pnl = pd.Series(0.0, index=ret.index)
    events = []

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

        if "CVS" not in window_ret.columns or "DOC" not in window_ret.columns:
            continue

        cvs_window = window_ret["CVS"].fillna(0)
        doc_window = window_ret["DOC"].fillna(0)

        # Long CVS, short DOC pair
        event_pnl = cvs_window - doc_window

        already_in = (pnl.iloc[p:end_idx] != 0).any()
        if not already_in:
            pnl.iloc[p:end_idx] = event_pnl.values

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

    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="HRSA 340B Surge Long CVS Short DOC")
    save_result(sid, m, extra={
        "rule": "Long CVS, short DOC (1:1) for 50 days when HRSA OPAIS quarterly 340B child-site net registrations >1.5σ above 8Q mean",
        "mechanism": "340B program expansion increases prescription volume for pharmacy benefit managers (CVS) while diverting revenue from hospital chains (DOC)",
        "source": "HRSA OPAIS quarterly data; yfinance",
        "n_events": len(events),
        "events": events,
        "status": "ok",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
