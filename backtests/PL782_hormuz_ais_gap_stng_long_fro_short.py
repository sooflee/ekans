"""PL782_hormuz_ais_gap_stng_long_fro_short — Strait-of-Hormuz AIS Gap -> Long STNG / Short FRO Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL782_hormuz_ais_gap_stng_long_fro_short"

    # Known Hormuz security incident dates (from implementation_notes)
    event_dates = pd.to_datetime([
        "2019-06-13",  # Gulf of Oman tanker attacks, twin-incident
        "2019-07-19",  # IRGC seized Stena Impero, Hormuz closure threat
        "2024-01-10",  # Houthi MR attacks, Red Sea/Hormuz rerouting
        "2024-04-13",  # IRGC seized MSC Aries, Hormuz gap cluster
    ])

    tickers = ["STNG", "FRO", "SPY"]
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "STNG" not in ret.columns or "FRO" not in ret.columns:
        return mark_failed(sid, f"missing required tickers; available: {list(ret.columns)}")

    spy_r = ret["SPY"]
    stng_r = ret["STNG"]
    fro_r = ret["FRO"]
    hold_days = 45  # calendar days, approx 31 trading days

    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for evt_date in event_dates:
        # Find T+1 trading day after event
        future_idx = ret.index[ret.index > evt_date]
        if len(future_idx) < 5:
            continue
        entry_date = future_idx[0]
        entry_pos = ret.index.get_loc(entry_date)

        # Approx 45 cal days ~ 31 trading days
        exit_pos = min(entry_pos + 31, len(ret) - 1)
        exit_date = ret.index[exit_pos]

        # Long STNG, Short FRO equal dollar notional pair
        # Pair P&L = STNG_return - FRO_return (equal dollar)
        stng_window = stng_r.iloc[entry_pos:exit_pos]
        fro_window = fro_r.iloc[entry_pos:exit_pos]
        spy_window = spy_r.iloc[entry_pos:exit_pos]

        pair_ret = stng_window.values[:exit_pos - entry_pos] - fro_window.values[:exit_pos - entry_pos]
        pnl.iloc[entry_pos:exit_pos] += pair_ret

        stng_cum = float((1 + stng_window).prod() - 1)
        fro_cum = float((1 + fro_window).prod() - 1)
        pair_cum = stng_cum - fro_cum
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "stng_return": round(stng_cum, 4),
            "fro_return": round(fro_cum, 4),
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Hormuz AIS Gap -> Long STNG / Short FRO Pair")
    save_result(sid, m, extra={
        "rule": "Long STNG / Short FRO equal-dollar pair at T+1 after Hormuz security incident; hold 45 days or until advisory lifted",
        "mechanism": "Hormuz incident forces MR/LR2 tankers to Cape rerouting (+20% ton-miles) benefiting STNG; suppresses VLCC AG-Asia utilization hurting FRO",
        "source": "UKHO MARLO bulletins; Lloyd's JWC; 4 known events 2019-2024",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}, CAGR={m.get('cagr', 'N/A')}")


if __name__ == "__main__":
    main()
