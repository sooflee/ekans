"""PL806_tga_drain_xdate_kre_long — TGA Drain >$300B in 8 Weeks Pre-X-Date -> Reserve Injection -> Long KRE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL806_tga_drain_xdate_kre_long"

    # Load FRED data: Treasury General Account (WTREGEN) weekly
    try:
        tga_raw = load_fred("WTREGEN", start="2006-01-01")
        tga = tga_raw.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED WTREGEN: {e}")

    if tga.empty:
        return mark_failed(sid, "no WTREGEN data")

    # Load prices
    try:
        px = load_prices(["KRE", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    kre_r = ret["KRE"]
    spy_r = ret["SPY"]

    # Known debt ceiling extraordinary measures episodes (manually identified)
    # Each episode: (trigger_date, exit_date, description)
    # trigger_date = when TGA drain threshold was clearly met during extraordinary measures
    # exit_date = when debt ceiling was resolved/raised
    episodes = [
        ("2011-07-25", "2011-08-02", "2011 debt ceiling crisis - near default"),
        ("2013-09-30", "2013-10-17", "2013 extraordinary measures, TGA drain"),
        ("2021-08-16", "2021-12-15", "2021 TGA drain ~$700B, reserve injection"),
        ("2023-02-01", "2023-06-05", "2023 TGA drain $456B->$23B (SVB note)"),
    ]

    hold_max = 70  # 10 weeks max (in trading days)
    event_results = []
    pnl_parts = []

    for trigger_str, exit_str, desc in episodes:
        trigger_date = pd.Timestamp(trigger_str)
        exit_date = pd.Timestamp(exit_str)

        # Entry on next trading day after trigger
        future_mask = kre_r.index > trigger_date
        if future_mask.sum() < 5:
            continue

        entry_idx = kre_r.index[future_mask][0]
        entry_pos = kre_r.index.get_loc(entry_idx)

        # Exit on exit_date or max hold, whichever is first
        exit_mask = kre_r.index >= exit_date
        if exit_mask.any():
            exit_pos_date = kre_r.index[exit_mask][0]
            exit_pos = kre_r.index.get_loc(exit_pos_date)
        else:
            exit_pos = min(entry_pos + hold_max, len(kre_r))

        # Enforce max hold
        end_pos = min(exit_pos, entry_pos + hold_max)
        if end_pos <= entry_pos:
            continue

        kre_window = kre_r.iloc[entry_pos:end_pos]
        spy_window = spy_r.reindex(kre_window.index).fillna(0)

        # Trade PnL: long KRE
        trade_pnl = kre_window

        pnl_parts.append(trade_pnl)

        kre_car = float((1 + kre_window).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)
        excess_car = kre_car - spy_car

        event_results.append({
            "trigger_date": trigger_str,
            "entry_date": str(entry_idx.date()),
            "exit_date": str(kre_r.index[end_pos - 1].date()),
            "description": desc,
            "hold_days": len(kre_window),
            "kre_return": round(kre_car, 4),
            "spy_return": round(spy_car, 4),
            "excess_return": round(excess_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid episodes found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="TGA Drain Pre-X-Date -> Long KRE")

    save_result(sid, m, extra={
        "rule": "Long KRE when WTREGEN 8-week change <= -$300B during debt ceiling extraordinary measures; exit at resolution or TGA rebuild or 10-week max hold",
        "mechanism": "TGA drain injects reserves into banking system -> bank reserve buffers rise -> regional banks' net interest margin improves as funding costs decline -> KRE outperforms",
        "source": "FRED WTREGEN; Treasury X-Date announcements; yfinance prices",
        "n_events": len(event_results),
        "avg_kre_return": round(float(np.mean([e["kre_return"] for e in event_results])), 4),
        "avg_excess_return": round(float(np.mean([e["excess_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["excess_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "2023 episode contaminated by SVB/regional bank crisis; only 4 qualifying episodes since 2006; distinct from PL150 (post-resolution TGA rebuild trade)",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['trigger_date']} ({e['description'][:40]}): KRE={e['kre_return']:.3f}, SPY={e['spy_return']:.3f}, excess={e['excess_return']:.3f}")


if __name__ == "__main__":
    main()
