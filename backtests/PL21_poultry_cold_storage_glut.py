"""PL21_poultry_cold_storage_glut — Poultry Cold Storage Glut → Short TSN+PPC
When USDA Cold Storage shows frozen poultry >15% above 5yr avg,
short TSN+PPC for 60 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL21_poultry_cold_storage_glut"
    try:
        px = load_prices(["TSN", "PPC", "SPY"], start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded USDA Cold Storage glut months (frozen poultry >15% above 5yr avg)
    # Dates = approximate report release date (~22nd of month)
    glut_events = [
        ("2017-02-22", "Q4 2016 / Q1 2017 production overshoot, record frozen stocks"),
        ("2018-08-22", "Trade war with China cut poultry exports, inventory piled up"),
        ("2019-02-22", "Lingering trade war oversupply, weak export demand"),
        ("2022-08-22", "Post-COVID production ramp overshot demand recovery"),
        ("2024-02-22", "Efficiency gains + weak China export demand"),
    ]

    events = []
    pnl_parts = []
    hold_days = 60

    for date_str, desc in glut_events:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        # Build basket — both TSN and PPC should be available
        available = []
        for t in ["TSN", "PPC"]:
            if t in ret.columns and not ret[t].iloc[window].isna().all():
                available.append(t)
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        # Short = negative of basket return
        short_pnl = -basket_r
        spy_window = spy_r.iloc[window]
        pnl_parts.append(short_pnl)

        short_cum = float((1 + short_pnl).prod() - 1)
        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "report_date": date_str,
            "description": desc,
            "basket_60d_return": round(basket_cum, 4),
            "short_return": round(short_cum, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess_vs_spy": round(short_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No glut events with price data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Poultry Glut Short TSN+PPC")

    avg_short = np.mean([e["short_return"] for e in events])
    win_count = sum(1 for e in events if e["short_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Short TSN+PPC equal-weight for 60 days when USDA Cold Storage frozen poultry >15% above 5yr avg",
        "mechanism": "Frozen poultry glut → storage cost pressure → processor liquidation at discount → spot chicken prices fall → margin compression",
        "source": "USDA Cold Storage (hand-coded glut months); yfinance",
        "n_events": len(events),
        "avg_short_return": round(avg_short, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg short return: {avg_short*100:.1f}%  Win rate: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["short_return"] > 0 else "-"
        print(f"  {flag} {e['report_date']}: short {e['short_return']*100:+.1f}%, spy {e['spy_60d_return']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
