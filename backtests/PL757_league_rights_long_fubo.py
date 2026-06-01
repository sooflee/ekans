"""PL757_league_rights_long_fubo — MLB/NBA Rights Renewal Calendar -> Long FUBO Pre-Announcement"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL757_league_rights_long_fubo"

    # Hand-coded MLB/NBA streaming-rights renewal calendar (known from trade press)
    # FUBO IPO: Oct 2020, so events from 2021 onward
    # Each tuple: (approximate deadline date, label)
    # Strategy: 60 trading days before each deadline, go long FUBO, hold to deadline
    events_raw = [
        # MLB streaming rights renewal discussions 2021-2022
        ("2022-01-15", "MLB streaming rights renewal 2022"),
        # NBA media rights renewal 2024 Q2 (known event in strat)
        ("2024-06-30", "NBA media rights renewal 2024-Q2"),
        # MLB Apple TV+ renewal discussions 2023
        ("2023-03-01", "MLB Apple TV+ 2023"),
        # NHL / additional league discussions 2022-2023
        ("2023-06-30", "NHL rights renewal 2023"),
        # ESPN/ABC NBA deal renegotiation 2025
        ("2025-02-01", "NBA ESPN renegotiation 2025"),
    ]

    try:
        px = load_prices(["FUBO", "SPY"], start="2020-10-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "FUBO" not in px.columns:
        return mark_failed(sid, "FUBO price data unavailable")

    ret = daily_returns(px)
    fubo_r = ret["FUBO"].dropna()
    spy_r = ret["SPY"].dropna()

    if len(fubo_r) < 60:
        return mark_failed(sid, f"insufficient FUBO data: {len(fubo_r)} days")

    hold_days = 60  # 60 trading days
    pnl = pd.Series(0.0, index=fubo_r.index)
    evts = []

    for deadline_str, label in events_raw:
        deadline = pd.Timestamp(deadline_str)

        # Find the entry point: 60 trading days before the deadline
        # Get all trading days up to (and including) the deadline
        before_deadline = fubo_r.index[fubo_r.index <= deadline]
        if len(before_deadline) < hold_days:
            continue  # Not enough history before this deadline

        # Entry = 60 trading days before the deadline date
        entry_idx = before_deadline[-hold_days]
        entry_pos = fubo_r.index.get_loc(entry_idx)

        # Exit: up to deadline or end of data
        exit_candidates = fubo_r.index[fubo_r.index >= deadline]
        if len(exit_candidates) == 0:
            exit_pos = len(fubo_r)
        else:
            exit_idx = exit_candidates[0]
            exit_pos = fubo_r.index.get_loc(exit_idx)

        window = fubo_r.iloc[entry_pos:exit_pos + 1]
        if len(window) == 0:
            continue

        pnl.iloc[entry_pos:exit_pos + 1] = window.values

        # Compute event-level stats
        cum_ret = float((1 + window).prod() - 1)
        spy_window = spy_r.reindex(window.index)
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else None

        evts.append({
            "deadline": deadline_str,
            "label": label,
            "entry_date": str(entry_idx.date()),
            "n_days": len(window),
            "fubo_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events processed: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    # Use only days with active positions
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="MLB/NBA Rights Renewal -> Long FUBO")
    returns_list = [e["fubo_return"] for e in evts]

    save_result(sid, m, extra={
        "rule": "Long FUBO 60 trading days before MLB/NBA streaming-rights renewal deadline",
        "mechanism": "Pre-announcement drift: sports streaming rights renewals drive expectations of subscriber growth and revenue for vMVPD platforms like FUBO",
        "source": "Hand-coded calendar from trade press; yfinance FUBO, SPY",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")
    for e in evts:
        print(f"  {e['label']}: FUBO={e['fubo_return']:.1%}, SPY={e.get('spy_return', 'N/A')}")


if __name__ == "__main__":
    main()
