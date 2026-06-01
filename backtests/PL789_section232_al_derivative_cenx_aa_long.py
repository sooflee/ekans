"""PL789_section232_al_derivative_cenx_aa_long — Section 232 Aluminum Derivative Expansion -> Long CENX / AA"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL789_section232_al_derivative_cenx_aa_long"

    # Federal Register Section 232 aluminum proclamation publication dates
    event_dates = pd.to_datetime([
        "2018-03-08",  # Proclamation 9704 - original 10% aluminum tariff
        "2020-01-24",  # Proclamation 9980 - first derivative expansion to foil/wire/cable
        "2025-02-14",  # Trump restoration + derivatives expansion
    ])

    tickers = ["CENX", "AA", "KALU", "SPY"]
    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["CENX", "AA"] if t in ret.columns]
    if not available:
        return mark_failed(sid, f"no primary tickers available; got: {list(ret.columns)}")

    spy_r = ret["SPY"]
    hold_days = 56  # 8 weeks ~ 56 calendar days ~ 40 trading days

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

        basket_ret = ret[available].iloc[entry_pos:exit_pos].mean(axis=1)
        spy_window = spy_r.iloc[entry_pos:exit_pos]
        n = exit_pos - entry_pos

        pnl.iloc[entry_pos:exit_pos] += basket_ret.values[:n]

        basket_cum = float((1 + basket_ret).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        per_ticker = {}
        for t in available:
            tw = ret[t].iloc[entry_pos:exit_pos]
            per_ticker[f"{t}_return"] = round(float((1 + tw).prod() - 1), 4)

        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "basket_return": round(basket_cum, 4),
            "spy_return": round(spy_cum, 4),
            "alpha": round(basket_cum - spy_cum, 4),
            **per_ticker,
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Section 232 Al Derivative Expansion -> Long CENX/AA")
    save_result(sid, m, extra={
        "rule": "Long CENX/AA equal-weight at T+1 after Federal Register publishes Section 232 aluminum derivative-article expansion proclamation; hold 8 weeks or until suspension/revocation",
        "mechanism": "Tariff on derivative articles (extrusions, foil, sheet) raises domestic Midwest Premium; primary smelters CENX and AA capture the full premium uplift in spot contracts and forward book repricing",
        "source": "Federal Register API; Proclamation 9704 (Mar 2018), 9980 (Jan 2020), Feb 2025 restoration; 3 events",
        "n_events": len(events),
        "available_tickers": available,
        "events": events,
        "note": "2020 event may be noisy due to COVID timing (event Jan 24, crash Feb 24)",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}, CAGR={m.get('cagr', 'N/A')}")


if __name__ == "__main__":
    main()
