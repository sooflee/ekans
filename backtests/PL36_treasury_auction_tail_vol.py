"""PL36_treasury_auction_tail_vol — Treasury Auction Tail >3bps -> Long VIXY
Hand-code auctions that tailed >3bps. Long VIXY 20 trading days.
Events: 2023-10-12, 2023-11-09, 2024-01-11, 2025-01-08.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL36_treasury_auction_tail_vol"
    try:
        px = load_prices(["VIXY", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded treasury auction tail events >3bps
    tail_events = [
        ("2023-10-12", "30Y auction tail 5.3bps — historic weak demand"),
        ("2023-11-09", "30Y auction tail 3.7bps — continued supply pressure"),
        ("2024-01-11", "10Y auction tail 3.1bps — January refunding stress"),
        ("2025-01-08", "10Y auction tail 3.6bps — rate uncertainty"),
    ]

    events = []
    pnl_parts = []
    hold_days = 20  # ~1 month

    for date_str, desc in tail_events:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        if "VIXY" not in ret.columns or ret["VIXY"].iloc[window].isna().all():
            continue

        vixy_r = ret["VIXY"].iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(vixy_r)

        vixy_cum = float((1 + vixy_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "entry_date": date_str,
            "description": desc,
            "vixy_20d_return": round(vixy_cum, 4),
            "spy_20d_return": round(spy_cum, 4),
            "excess": round(vixy_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No auction tail events with VIXY data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Treasury Auction Tail -> Long VIXY")

    avg_vixy = np.mean([e["vixy_20d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["vixy_20d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long VIXY 20d when 10Y/30Y auction tails >3bps",
        "mechanism": "Auction tail signals weak demand for Treasuries -> rate vol spikes -> VIX rises as rate uncertainty transmits to equities",
        "source": "Treasury auction results (hand-coded tail events); yfinance VIXY",
        "n_events": len(events),
        "avg_vixy_return": round(avg_vixy, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg VIXY: {avg_vixy*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["vixy_20d_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']}: VIXY {e['vixy_20d_return']*100:+.1f}%, SPY {e['spy_20d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
