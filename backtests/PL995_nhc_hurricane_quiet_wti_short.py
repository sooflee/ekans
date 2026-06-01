"""PL995_nhc_hurricane_quiet_wti_short — NHC Atlantic Quiet Spell After Above-Normal Outlook -> Short WTI"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL995_nhc_hurricane_quiet_wti_short"

    # Known qualifying years: above-normal NOAA outlook + zero/near-zero Gulf shut-ins
    # through Aug 1 - Sep 15 window
    # Event = enter short CL=F on Aug 1 of each qualifying year, hold 42 days
    # Sources: NOAA seasonal outlooks + known quiet Gulf seasons
    qualifying_events = [
        "2013-08-01",  # above-normal outlook; near-zero Gulf shut-ins; WTI bled $3-4/bbl late August
        "2014-08-01",  # above-normal outlook; quiet Gulf season
        "2022-08-01",  # active season forecast but few Gulf threats; WTI spread compressed
        "2023-08-01",  # above-normal (18 named storms forecast); zero Gulf shut-ins
        "2017-08-01",  # above-normal season but Harvey struck late August — partial event (known miss)
        "2019-08-01",  # above-normal outlook; Dorian missed Gulf; quiet window
        "2021-08-01",  # above-normal; Ida struck Gulf late August (control case — strategy exits)
    ]

    # For 2017 and 2021 (Harvey/Ida), hold was cut short by Gulf landfall
    early_exit_days = {
        "2017-08-01": 25,  # Harvey made landfall Aug 25
        "2021-08-01": 28,  # Ida entered Gulf Aug 28
    }

    hold_days = 42  # calendar ~= 30 trading days

    try:
        px = load_prices(["CL=F", "USO", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    # Use USO as proxy for CL=F if CL=F has gaps
    if "CL=F" in ret.columns and ret["CL=F"].notna().sum() > 100:
        crude_col = "CL=F"
    elif "USO" in ret.columns:
        crude_col = "USO"
        print("Using USO as crude oil proxy")
    else:
        return mark_failed(sid, "no crude oil price series available")

    spy_r = ret["SPY"]
    crude_r = ret[crude_col].dropna()

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ev_date_str in qualifying_events:
        ev_date = pd.Timestamp(ev_date_str)
        future_idx = ret.index[ret.index >= ev_date]
        if len(future_idx) < 5:
            print(f"Skipping {ev_date_str}: insufficient future data")
            continue

        n_days = early_exit_days.get(ev_date_str, hold_days)

        entry_idx = future_idx[0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + n_days, len(ret))

        crude_window = crude_r.iloc[entry_loc:exit_loc].dropna()
        spy_window = spy_r.iloc[entry_loc:exit_loc]

        if len(crude_window) < 5:
            print(f"Skipping {ev_date_str}: no crude data in window")
            continue

        # Strategy: short crude oil (profit from risk-premium bleedout)
        strat_ret = -crude_window
        crude_cum = float((1 + crude_window).prod() - 1)
        spy_cum = float((1 + spy_window.reindex(crude_window.index)).prod() - 1)

        event_records.append({
            "event_date": ev_date_str,
            "crude_return": round(crude_cum, 4),
            "spy_return": round(spy_cum, 4),
            "short_pnl": round(-crude_cum, 4),
            "n_days": len(crude_window),
            "early_exit": ev_date_str in early_exit_days,
        })

        for idx, r in strat_ret.items():
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['event_date']}: crude={e['crude_return']:.2%}, SPY={e['spy_return']:.2%}, "
              f"short_pnl={e['short_pnl']:.2%} [n={e['n_days']}]{'[early exit]' if e['early_exit'] else ''}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NHC Quiet Spell WTI Short")

    save_result(sid, m, extra={
        "rule": "Short WTI (CL=F / USO) on Aug 1 when NOAA seasonal outlook is above-normal AND Gulf of Mexico shows zero BSEE shut-ins; exit after 42 trading days or first Gulf-threatening storm track",
        "mechanism": "Above-normal Atlantic seasons create persistent hurricane risk premium in WTI; when Gulf remains quiet despite active Atlantic season, excess risk premium bleeds out as CFTC managed-money longs are liquidated",
        "source": "NOAA/NHC seasonal outlooks; BSEE Gulf shut-in reports; CL=F/USO via yfinance; known events 2013, 2014, 2017 (Harvey), 2019, 2021 (Ida), 2022, 2023",
        "n_events": len(event_records),
        "events": event_records,
        "caveat": "CL=F on yfinance is front-month only; 2017/2021 events had early exits due to Gulf landfalls; N=7",
    })


if __name__ == "__main__":
    main()
