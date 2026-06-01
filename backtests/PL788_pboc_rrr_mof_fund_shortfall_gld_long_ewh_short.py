"""PL788_pboc_rrr_mof_fund_shortfall_gld_long_ewh_short — PBoC RRR Cut + MoF Fund Shortfall -> Long GLD / Short EWH"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL788_pboc_rrr_mof_fund_shortfall_gld_long_ewh_short"

    # Known PBoC RRR + MoF property fund paired announcement dates
    # (from known_events + implementation_notes)
    event_dates = pd.to_datetime([
        "2023-08-25",  # PBoC 25bps RRR cut + small property fund (market found insufficient)
        "2024-10-12",  # PBoC rate cut + PSL expansion, credibility questioned
    ])

    tickers = ["GLD", "EWH", "SPY"]
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "GLD" not in ret.columns or "EWH" not in ret.columns:
        return mark_failed(sid, f"missing required tickers; available: {list(ret.columns)}")

    spy_r = ret["SPY"]
    gld_r = ret["GLD"]
    ewh_r = ret["EWH"]
    hold_days = 56  # 8 weeks = ~56 calendar days ~ 40 trading days

    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for evt_date in event_dates:
        # Find T+1 trading day after event
        future_idx = ret.index[ret.index > evt_date]
        if len(future_idx) < 5:
            continue
        entry_date = future_idx[0]
        entry_pos = ret.index.get_loc(entry_date)

        # 8 weeks ~ 40 trading days
        exit_pos = min(entry_pos + 40, len(ret) - 1)
        exit_date = ret.index[exit_pos]

        gld_window = gld_r.iloc[entry_pos:exit_pos]
        ewh_window = ewh_r.iloc[entry_pos:exit_pos]
        spy_window = spy_r.iloc[entry_pos:exit_pos]
        n = exit_pos - entry_pos

        # Long GLD 40%, Short EWH 60%
        pair_daily = (0.40 * gld_window.values[:n] - 0.60 * ewh_window.values[:n])
        pnl.iloc[entry_pos:exit_pos] += pair_daily

        gld_cum = float((1 + gld_window).prod() - 1)
        ewh_cum = float((1 + ewh_window).prod() - 1)
        pair_cum = 0.40 * gld_cum - 0.60 * ewh_cum
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "gld_return": round(gld_cum, 4),
            "ewh_return": round(ewh_cum, 4),
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4),
            "alpha": round(pair_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 15:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="PBoC RRR Cut + MoF Fund Shortfall -> Long GLD / Short EWH")
    save_result(sid, m, extra={
        "rule": "Long GLD (40%) / Short EWH (60%) at T+1 after PBoC RRR cut + MoF property fund announcement where fund size covers <1/3 of IMF-estimated developer inventory hole and USDCNH > 7.20",
        "mechanism": "Credibility gap between announced rescue size and actual property-sector hole triggers CNH weakness and gold safe-haven demand; HK property developers in EWH face fundamental demand collapse",
        "source": "PBoC press releases; MoF/State Council announcements; IMF Article IV 2025; 2 known events 2023-2024",
        "n_events": len(events),
        "events": events,
        "note": "Only 2 qualifying events available; use caution with statistical conclusions",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}, CAGR={m.get('cagr', 'N/A')}")


if __name__ == "__main__":
    main()
