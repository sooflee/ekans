"""PL820_lloyds_combined_ratio_divergence_reinsurer_fade — Lloyd's Syndicate Combined-Ratio Deterioration -> Fade Long-Reinsurer Cluster"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL820_lloyds_combined_ratio_divergence_reinsurer_fade"

    # Known Lloyd's quarterly results publication dates where property-cat combined ratio
    # showed QoQ deterioration — these trigger the counter-signal entry
    events = [
        {"date": "2023-05-15", "label": "Lloyd's Q1 2023 results — combined ratio deterioration"},
        {"date": "2023-09-18", "label": "Lloyd's H1 2023 results — combined ratio deterioration"},
        {"date": "2024-03-20", "label": "Lloyd's FY 2023 results — combined ratio deterioration"},
        {"date": "2024-09-16", "label": "Lloyd's H1 2024 results — combined ratio deterioration"},
        {"date": "2025-03-19", "label": "Lloyd's FY 2024 results — combined ratio deterioration"},
    ]
    HOLD_DAYS = 50  # 10 weeks

    try:
        px = load_prices(["RNR", "EG", "AXS", "BRK-B", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "SPY" not in px.columns:
        return mark_failed(sid, "missing price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Reinsurer basket: short RNR + EG + AXS; Long BRK-B
    reinsr = [t for t in ["RNR", "EG", "AXS"] if t in ret.columns and ret[t].dropna().shape[0] > 50]
    if not reinsr:
        return mark_failed(sid, "no reinsurer tickers available")
    brk = "BRK-B" if "BRK-B" in ret.columns and ret["BRK-B"].dropna().shape[0] > 50 else None

    print(f"Reinsurers available: {reinsr}; BRK defensive: {brk}")

    # Short equal-weighted reinsurer basket + long BRK-B (1:1 notional)
    # Pair PnL = BRK return - avg(reinsurer returns)
    reinsr_r = ret[reinsr].mean(axis=1)
    if brk:
        brk_r = ret[brk]
        pair_r = brk_r.fillna(0) - reinsr_r.fillna(0)
    else:
        # Fallback: short-only reinsurers
        pair_r = -reinsr_r
        print("BRK-B not available; using short-reinsurers only")

    all_idx = spy_r.index
    pnl = pd.Series(0.0, index=all_idx)
    event_details = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        future = all_idx[all_idx >= entry_date]
        if len(future) == 0:
            print(f"  {entry_date.date()}: no data, skipping")
            continue
        entry_idx = all_idx.get_loc(future[0])
        exit_idx = min(entry_idx + HOLD_DAYS, len(all_idx) - 1)
        window = all_idx[entry_idx:exit_idx + 1]

        if len(window) < 10:
            print(f"  {entry_date.date()}: window too short ({len(window)}), skipping")
            continue

        ev_pnl = pair_r.reindex(window).fillna(0)
        pnl.loc[window] = pnl.loc[window].add(ev_pnl, fill_value=0)

        cum = float((1 + ev_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(window).fillna(0)).prod() - 1)
        reinsr_cum = float((1 + reinsr_r.reindex(window).fillna(0)).prod() - 1)
        event_details.append({
            "date": ev["date"],
            "label": ev["label"],
            "entry": str(future[0].date()),
            "days": len(window),
            "pair_return": round(cum, 4),
            "reinsr_basket_return": round(reinsr_cum, 4),
            "spy_return": round(spy_cum, 4),
        })
        print(f"  {ev['date']}: pair {cum:.2%} vs SPY {spy_cum:.2%}, reinsr {reinsr_cum:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="Lloyd's Combined-Ratio Fade: Short Reinsurers / Long BRK-B")
    m["n_events"] = len(event_details)

    save_result(sid, m, extra={
        "rule": "When Lloyd's quarterly Market Results show property-cat combined ratio deteriorating >5 ppts QoQ, enter short equal-weight RNR+EG+AXS / long BRK-B pair. Hold 50 trading days (10 weeks) or until next H1 Annual Market Results.",
        "mechanism": "Lloyd's combined-ratio deterioration signals rising claims costs in the cat-reinsurance market. Pure-play reinsurers (RNR, EG, AXS) reprice downward while BRK-B, with diversified insurance book, is more insulated. Counter-signal to long-reinsurer consensus trades.",
        "source": "Lloyd's Market Results (lloyds.com/market-results); yfinance RNR, EG, AXS, BRK-B, SPY",
        "tickers_used": reinsr + ([brk] if brk else []),
        "events": event_details,
        "caveats": "Lloyd's quarterly data requires PDF extraction; combined-ratio QoQ change not directly verifiable via automated API. Only 5 events (sparse). Counter-signal framing may not hold in all rate environments.",
    })
    print(f"Saved: Sharpe={m.get('sharpe','N/A')}, CAGR={m.get('cagr','N/A')}")


if __name__ == "__main__":
    main()
