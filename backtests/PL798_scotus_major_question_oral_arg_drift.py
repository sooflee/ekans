"""PL798_scotus_major_question_oral_arg_drift — SCOTUS Major-Question Oral-Argument Day -> Defendant Industry Drift"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL798_scotus_major_question_oral_arg_drift"

    # Known major-question doctrine SCOTUS oral argument events and target ETFs
    # Mapped manually per strategy spec:
    #   WV v. EPA (argued 2022-02-28) -> XLU (energy/utility sector)
    #   CFSA v. CFPB funding (argued 2023-10-03) -> KRE (banking)
    #   Loper Bright Enterprises v. Raimondo (argued 2024-01-17) -> XLI (labor/agency deference)
    #   Corner Post v. Board of Governors (argued 2024-02-20) -> KRE (financial regulation)
    events = [
        ("2022-02-28", "XLU", "WV v EPA - energy/utility deregulation"),
        ("2023-10-03", "KRE", "CFSA v CFPB - banking/consumer finance"),
        ("2024-01-17", "XLI", "Loper Bright - labor/agency deference"),
        ("2024-02-20", "KRE", "Corner Post - financial regulation"),
    ]

    # Load all needed tickers
    tickers = ["XLU", "KRE", "XLI", "SPY"]
    try:
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    hold = 15  # 15 trading days holding period
    event_results = []
    pnl_parts = []  # list of (index, values) for each event

    for arg_date_str, etf, case_name in events:
        arg_date = pd.Timestamp(arg_date_str)
        if etf not in ret.columns:
            continue
        etf_r = ret[etf]

        # Entry at T+1 (day after oral argument)
        future_mask = etf_r.index > arg_date
        if future_mask.sum() < hold:
            continue

        entry_idx = etf_r.index[future_mask][0]
        pos = etf_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold, len(etf_r))

        # Long 1x ETF, short 0.5x SPY (as per rule)
        etf_window = etf_r.iloc[pos:end_pos]
        spy_window = spy_r.reindex(etf_window.index).fillna(0)
        trade_pnl = etf_window - 0.5 * spy_window

        pnl_parts.append(trade_pnl)

        # Compute event-level stats
        etf_car = float((1 + etf_window).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)
        excess_car = etf_car - spy_car

        event_results.append({
            "arg_date": arg_date_str,
            "etf": etf,
            "case": case_name,
            "entry_date": str(entry_idx.date()),
            "hold_days": len(etf_window),
            "etf_return": round(etf_car, 4),
            "spy_return": round(spy_car, 4),
            "excess_return": round(excess_car, 4),
            "trade_pnl": round(float(trade_pnl.sum()), 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid events found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Build combined PnL series (no overlap between events given different dates)
    all_pnl = pd.concat(pnl_parts).sort_index()
    # Remove any duplicate index entries (overlapping windows) by keeping first
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    # Need at least 20 days of data
    if len(all_pnl) < 20:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="SCOTUS Major-Question Oral-Arg Drift")

    save_result(sid, m, extra={
        "rule": "T+1 after SCOTUS oral argument in major-question doctrine case: long defendant-industry ETF (XLU/KRE/XLI/XLB), short 0.5x SPY, hold T+15 trading days",
        "mechanism": "Uncertainty about regulatory scope resolves in favor of regulated industry after argument; market underreacts to judicial signals on oral argument day",
        "source": "SCOTUSblog/Oyez oral argument dates; yfinance prices",
        "n_events": len(event_results),
        "avg_excess_return": round(float(np.mean([e["excess_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["excess_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Only 4 events in modern major-question era; severely underpowered for statistical inference",
    })

    print(f"Done: {len(event_results)} events, avg excess return: {np.mean([e['excess_return'] for e in event_results]):.3f}")
    for e in event_results:
        print(f"  {e['arg_date']} {e['etf']} ({e['case']}): ETF={e['etf_return']:.3f}, SPY={e['spy_return']:.3f}, excess={e['excess_return']:.3f}")


if __name__ == "__main__":
    main()
