"""PL793 — Divided Government Lock-In (Polymarket >70%) -> Long TLT / Short SPY Fiscal Gridlock Counter-Signal"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL793_divided_govt_lockin_tlt_long_spy_short"
    # Historical midterm election events that produced divided government
    # 2018-11-06: Democrats flipped the House (divided Congress)
    # 2022-11-08: Narrow GOP House win (divided Congress)
    # Strategy: Long TLT (1x) + Short SPY (0.5x) from T-14 to T+28 around election day
    KNOWN_EVENTS = [
        {"date": "2018-11-06", "result": "Dem House flip 2018"},
        {"date": "2022-11-08", "result": "Narrow GOP House 2022"},
    ]
    HOLD_BEFORE = 14   # enter 14 trading days before election
    HOLD_AFTER = 28    # exit 28 trading days after election
    TLT_WEIGHT = 1.0
    SPY_WEIGHT = -0.5  # short SPY

    try:
        px = load_prices(["TLT", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "TLT" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing TLT or SPY price data")

    ret = daily_returns(px)
    tlt_r = ret["TLT"].dropna()
    spy_r = ret["SPY"].dropna()

    common_idx = tlt_r.index.intersection(spy_r.index)
    pnl = pd.Series(0.0, index=common_idx)
    event_details = []

    for ev in KNOWN_EVENTS:
        ev_date = pd.Timestamp(ev["date"])

        # Find election day index in price data
        future = common_idx[common_idx >= ev_date]
        if len(future) == 0:
            print(f"Event {ev_date.date()}: no price data on/after election date, skipping")
            continue

        election_idx = common_idx.get_loc(future[0])

        # Entry: T-14 trading days before election
        entry_pos = max(0, election_idx - HOLD_BEFORE)
        entry_date = common_idx[entry_pos]

        # Exit: T+28 trading days after election day
        exit_pos = min(election_idx + HOLD_AFTER, len(common_idx) - 1)
        window = common_idx[entry_pos:exit_pos + 1]

        if len(window) < 5:
            print(f"Event {ev_date.date()}: insufficient window, skipping")
            continue

        # Pair PnL: 1x Long TLT + 0.5x Short SPY
        pair_ret = (TLT_WEIGHT * tlt_r.reindex(window) +
                    SPY_WEIGHT * spy_r.reindex(window))

        pnl.loc[window] = pair_ret.values

        # Event metrics
        tlt_cum = float((1 + tlt_r.reindex(window)).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(window)).prod() - 1)
        pair_cum = float((1 + pair_ret).prod() - 1)

        event_details.append({
            "event_date": str(ev_date.date()),
            "entry_date": str(entry_date.date()),
            "result": ev["result"],
            "tlt_return": round(tlt_cum, 4),
            "spy_return": round(spy_cum, 4),
            "pair_return": round(pair_cum, 4),
            "days_held": len(window),
        })
        print(f"Event {ev_date.date()}: TLT {tlt_cum:.2%}, SPY {spy_cum:.2%}, pair {pair_cum:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 5:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="Divided Govt Lock-In → Long TLT / Short SPY")
    m["n_events"] = len(event_details)

    save_result(sid, m, extra={
        "rule": "Long TLT (1x) + Short SPY (0.5x) starting T-14 before midterm elections that produce divided government. Exit at T+28 post-election. Historical seeds: 2018 Dem House flip, 2022 narrow GOP House.",
        "mechanism": "Divided government creates fiscal gridlock (spending/deficit fights), reducing inflationary fiscal impulse, supporting long-duration Treasuries. Short SPY captures risk-off when deficit-expansion is off the table and earnings multiple contracts.",
        "source": "Midterm election results; yfinance TLT, SPY; Polymarket for live execution",
        "known_events": [e["date"] for e in KNOWN_EVENTS],
        "events": event_details,
        "forward_looking_note": "Primary live trigger uses Polymarket 'Divided government 2026' probability >70% for 3 consecutive days in T-30 window. 2018/2022 are pure event-study seeds.",
        "caveats": "Only 2 historical events. 2018 and 2022 diverged in outcome (TLT up vs down); fiscal gridlock thesis varies by macro backdrop. Very limited statistical power.",
    })
    print(f"Saved result: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A'):.2%}")


if __name__ == "__main__":
    main()
