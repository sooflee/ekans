"""PL786_nbs_property_iron_ore_miner_short — NBS 70-City New-Home Price YoY < -10% -> Short BHP/VALE/RIO"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL786_nbs_property_iron_ore_miner_short"

    # Known NBS 70-City YoY <= -10% breach dates (from known_events + implementation_notes)
    event_dates = pd.to_datetime([
        "2014-09-15",  # NBS 70-City YoY breached -10%, Sept 2014
        "2015-07-15",  # NBS 70-City YoY breached -10%, July 2015
        "2021-11-15",  # NBS 70-City YoY breached -10%, Nov 2021
        "2023-01-16",  # NBS 70-City YoY breached -10%, Jan 2023
    ])

    tickers = ["BHP", "VALE", "RIO", "SPY"]
    try:
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)

    # Load FRED iron ore data for additional context (not strictly needed for event-study)
    try:
        iron_ore = load_fred("PIORECRUSDM", start="2013-01-01")
        iron_ore = iron_ore.squeeze()
    except Exception as e:
        iron_ore = None
        print(f"Note: FRED iron ore data unavailable: {e}")

    available_tickers = [t for t in ["BHP", "VALE", "RIO"] if t in ret.columns]
    if len(available_tickers) < 2:
        return mark_failed(sid, f"insufficient tickers available: {available_tickers}")

    spy_r = ret["SPY"]
    hold_days = 84  # 12 weeks = ~84 calendar days ~ 60 trading days

    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for evt_date in event_dates:
        # Find T+1 trading day after event
        future_idx = ret.index[ret.index > evt_date]
        if len(future_idx) < 5:
            continue
        entry_date = future_idx[0]
        entry_pos = ret.index.get_loc(entry_date)

        # 12 weeks ~ 60 trading days
        exit_pos = min(entry_pos + 60, len(ret) - 1)
        exit_date = ret.index[exit_pos]

        # Equal-weight SHORT BHP, VALE, RIO
        # Short = negative of returns
        basket_ret = -ret[available_tickers].iloc[entry_pos:exit_pos].mean(axis=1)
        spy_window = spy_r.iloc[entry_pos:exit_pos]

        pnl.iloc[entry_pos:exit_pos] += basket_ret.values[:exit_pos - entry_pos]

        short_cum = float((1 + basket_ret).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        # Per-ticker returns (long basis, for reporting)
        per_ticker = {}
        for t in available_tickers:
            tk_window = ret[t].iloc[entry_pos:exit_pos]
            per_ticker[f"{t}_return_long"] = round(float((1 + tk_window).prod() - 1), 4)

        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "short_basket_return": round(short_cum, 4),
            "spy_return": round(spy_cum, 4),
            "alpha": round(short_cum - spy_cum, 4),
            **per_ticker,
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NBS Property Collapse -> Short BHP/VALE/RIO")
    save_result(sid, m, extra={
        "rule": "Short equal-weight BHP/VALE/RIO at T+1 after NBS 70-City New-Home Price YoY breaches -10%; hold 12 weeks or until YoY recovers above -5%",
        "mechanism": "China property market collapse suppresses rebar/HRC demand, iron ore imports fall, miners face earnings downgrades; equity derating follows lagged price signal",
        "source": "NBS 70-City index (monthly release); FRED PIORECRUSDM; 4 known events 2014-2023",
        "n_events": len(events),
        "available_tickers": available_tickers,
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}, CAGR={m.get('cagr', 'N/A')}")


if __name__ == "__main__":
    main()
