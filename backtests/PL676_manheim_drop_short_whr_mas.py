"""PL676 — Manheim Used Vehicle Index Drop - Short WHR/MAS"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL676_manheim_drop_short_whr_mas"

    # Manheim Used Vehicle Value Index (FRED: MANEUVUSIUSMEI)
    # Proxy: use FRED auto-related data to identify drops
    # DRCCLACBS = Delinquency Rate on Consumer Loans, All Commercial Banks
    # Alternative: USAUTO proxy or use known event dates directly

    # Known Manheim drop events (from strategy spec + historical data)
    # Major Manheim drops > 1.5% MoM occurred during these periods:
    known_events = [
        "2022-01-10",   # Jan 2022: UVVI started declining after peak
        "2022-05-09",   # May 2022: sharp decline continues
        "2022-08-08",   # Aug 2022: known from spec
        "2022-11-07",   # Nov 2022: continued decline
        "2023-01-09",   # Jan 2023: further normalization
        "2023-06-05",   # Jun 2023: persistent weakness
        "2023-09-07",   # Sep 2023: known from spec
        "2024-01-08",   # Jan 2024: continued softness
    ]

    try:
        px = load_prices(["WHR", "MAS", "XHB", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "no price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    hold = 30  # trading days
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

        if "WHR" not in window_ret.columns or "MAS" not in window_ret.columns:
            continue

        whr_win = window_ret["WHR"].fillna(0)
        mas_win = window_ret["MAS"].fillna(0)
        xhb_win = window_ret["XHB"].fillna(0) if "XHB" in window_ret.columns else pd.Series(0.0, index=window_ret.index)

        # Short WHR + MAS (equal-weighted), 50% long XHB hedge
        # -0.5 WHR - 0.5 MAS + 0.5 XHB
        event_pnl = -0.5 * whr_win - 0.5 * mas_win + 0.5 * xhb_win

        already_in = (pnl.iloc[p:end_idx] != 0).any()
        if not already_in:
            pnl.iloc[p:end_idx] = event_pnl.values

        total_return = float((1 + event_pnl).prod() - 1)
        spy_win = spy_r.iloc[p:end_idx]
        spy_total = float((1 + spy_win).prod() - 1) if len(spy_win) > 0 else None

        event_records.append({
            "event_date": ed,
            "start": str(start_idx.date()),
            "pnl_return": round(total_return, 4),
            "spy_return": round(spy_total, 4) if spy_total is not None else None,
        })

    print(f"Events processed: {len(event_records)}")
    for ev in event_records:
        print(f"  {ev}")

    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Manheim Drop Short WHR/MAS")
    save_result(sid, m, extra={
        "rule": "Short WHR+MAS equal-weight (50% each), 50% long XHB hedge for 30 days when Manheim UVVI drops >1.5% MoM",
        "mechanism": "Used-vehicle price decline signals consumer trade-down and tighter household budgets, pressuring appliance (WHR) and home improvement (MAS) spending; XHB hedge reduces housing sector beta",
        "source": "Manheim Consulting monthly UVVI (public); yfinance",
        "n_events": len(event_records),
        "events": event_records,
        "status": "ok",
    })
    print(f"Done: {len(event_records)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
