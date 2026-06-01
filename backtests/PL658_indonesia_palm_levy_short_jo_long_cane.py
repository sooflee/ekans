"""PL658 Indonesia Palm Oil Export Levy Hike - Short JO Long CANE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL658_indonesia_palm_levy_short_jo_long_cane"

    # Indonesia Ministry of Trade palm oil export levy/DMO event dates
    # Per strategy spec: known events + historical levy changes 2020-2024
    # Major regulatory events that disrupted palm oil supply:
    events_str = [
        "2020-07-06",   # Indonesia implemented domestic market obligation (DMO)
        "2021-11-03",   # export levy raised amid surging palm oil prices
        "2022-03-07",   # export ban threat / DMO tightened (pre-ban announcement)
        "2022-04-28",   # known event: export levy hike (spec)
        "2023-06-01",   # known event: levy/DMO policy change (spec)
        "2024-02-12",   # biofuel mandate expansion increasing domestic absorption
    ]

    try:
        px = load_prices(["JO", "CANE", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    jo_r = ret["JO"]
    cane_r = ret["CANE"]
    spy_r = ret["SPY"]

    # Long CANE, short JO → net = CANE - JO (equal notional)
    pair_r = cane_r - jo_r

    hold = 30  # 30 trading days
    pnl = pd.Series(0.0, index=pair_r.index)
    events = []

    for ev_str in events_str:
        ev_dt = pd.Timestamp(ev_str)
        mask = pair_r.index >= ev_dt
        if mask.sum() < hold:
            continue
        ei = pair_r.index[mask][0]
        p = pair_r.index.get_loc(ei)
        ep = min(p + hold, len(pair_r))
        window = pair_r.iloc[p:ep]
        # avoid overlap
        if (pnl.iloc[p:ep] != 0).any():
            continue
        pnl.iloc[p:ep] = window.values[: ep - p]

        jo_fwd = float((1 + jo_r.iloc[p:ep]).prod() - 1)
        cane_fwd = float((1 + cane_r.iloc[p:ep]).prod() - 1)
        events.append({
            "event_date": ev_str,
            "jo_return": round(jo_fwd, 4),
            "cane_return": round(cane_fwd, 4),
            "trade_pnl": round(float((1 + window).prod() - 1), 4),
        })

    active = pnl[pnl != 0]
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(pnl, benchmark=spy_r,
                        name="Indonesia Palm Levy → Long CANE / Short JO")
    save_result(sid, m, extra={
        "rule": ("On Indonesia Ministry of Trade palm oil export levy hike or "
                 "DMO tightening announcement, go long CANE and short JO "
                 "equal notional for 30 trading days."),
        "mechanism": ("Indonesia controls ~60% of global palm oil exports. "
                      "Levy hikes restrict palm oil supply, raising vegetable "
                      "oil prices broadly (CANE/sugar benefits as substitution), "
                      "while OJ (JO) tends to diverge or suffer demand shifts."),
        "source": ("Indonesia Ministry of Trade decrees (peraturan.go.id); "
                   "yfinance: JO, CANE, SPY"),
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, {len(active)} active PnL days")


if __name__ == "__main__":
    main()
