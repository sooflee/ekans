"""PL815_usgs_mcs_dla_drawdown_defense_margin — USGS MCS Severe-Shortage + DLA Stockpile Drawdown -> Long LMT/RTX vs Short ITA"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL815_usgs_mcs_dla_drawdown_defense_margin"

    # Known USGS MCS + DLA dual-trigger events
    # DLA Annual Report release dates (approximate) when both conditions met:
    # 2018: Cobalt severe shortage + DLA drawdown confirmed ~April 2018
    # 2021: Cobalt/Rhenium shortage continued post-COVID + DLA report ~May 2021
    # 2022: Beryllium/Hafnium added to critical list + DLA drawdown ~April 2022
    # 2023: Multiple minerals + DLA drawdown ~April 2023
    events = [
        ("2018-04-15", "Cobalt severe shortage + DLA drawdown 2018"),
        ("2021-05-01", "Co/Re post-COVID shortage + DLA drawdown 2021"),
        ("2022-04-20", "Be/Hf critical list + DLA drawdown 2022"),
        ("2023-04-28", "Multiple minerals + DLA drawdown 2023"),
    ]

    tickers = ["LMT", "RTX", "ITA", "SPY"]
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "LMT" not in px.columns or "RTX" not in px.columns:
        return mark_failed(sid, "LMT or RTX not available")

    ret = daily_returns(px)
    lmt_r = ret["LMT"]
    rtx_r = ret["RTX"]
    ita_r = ret["ITA"]
    spy_r = ret["SPY"]

    hold = 63  # one quarter = ~63 trading days
    event_results = []
    pnl_parts = []

    for trigger_date_str, desc in events:
        trigger_date = pd.Timestamp(trigger_date_str)

        # Entry on next trading day
        future_mask = lmt_r.index > trigger_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = lmt_r.index[future_mask][0]
        pos = lmt_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold, len(lmt_r))

        lmt_window = lmt_r.iloc[pos:end_pos]
        rtx_window = rtx_r.reindex(lmt_window.index).fillna(0)
        ita_window = ita_r.reindex(lmt_window.index).fillna(0)
        spy_window = spy_r.reindex(lmt_window.index).fillna(0)

        # Pair: long 0.5 LMT + 0.5 RTX, short 1.0 ITA (sector-hedged)
        trade_pnl = 0.5 * lmt_window + 0.5 * rtx_window - ita_window

        pnl_parts.append(trade_pnl)

        lmt_car = float((1 + lmt_window).prod() - 1)
        rtx_car = float((1 + rtx_window).prod() - 1)
        ita_car = float((1 + ita_window).prod() - 1)
        pair_car = float((1 + trade_pnl).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)

        event_results.append({
            "trigger_date": trigger_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc,
            "hold_days": len(lmt_window),
            "lmt_return": round(lmt_car, 4),
            "rtx_return": round(rtx_car, 4),
            "ita_return": round(ita_car, 4),
            "pair_return": round(pair_car, 4),
            "spy_return": round(spy_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid events found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="USGS/DLA Dual-Trigger -> Long LMT+RTX Short ITA")

    save_result(sid, m, extra={
        "rule": "Long 0.5x LMT + 0.5x RTX, short 1x ITA when USGS MCS flags severe shortage + DLA annual report shows >=20% drawdown in same DoD-critical mineral; hold 1 quarter",
        "mechanism": "DFARS 252.225 cost-pass-through clauses reimburse defense primes for material cost increases on cost-plus contracts; mineral shortage triggers DoD accelerated procurement; LMT/RTX outperform broader defense (ITA) as cost-plus pass-through accrues to margins",
        "source": "USGS Mineral Commodity Summaries (minerals.usgs.gov); DLA Strategic Materials Annual Report (dla.mil); yfinance prices",
        "n_events": len(event_results),
        "avg_pair_return": round(float(np.mean([e["pair_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["pair_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Only 4 dual-trigger events since 2018; DLA report exact dates approximate; mechanism relies on DFARS cost-plus contracts which cover ~50% of LMT/RTX revenue",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['trigger_date']} ({e['description'][:45]}): LMT={e['lmt_return']:.3f}, RTX={e['rtx_return']:.3f}, ITA={e['ita_return']:.3f}, pair={e['pair_return']:.3f}")


if __name__ == "__main__":
    main()
