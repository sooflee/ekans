"""PL781_lloyds_jwc_listed_area_reinsurer_long — Lloyd's JWC Listed-Area Expansion -> Long P&C Reinsurer Basket"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL781_lloyds_jwc_listed_area_reinsurer_long"

    # Known JWC bulletin dates (hardcoded per implementation_notes)
    event_dates = pd.to_datetime([
        "2019-05-13",  # Fujairah tanker sabotage, JWC Gulf of Oman area
        "2019-09-14",  # Aramco Abqaiq attack, JWC Saudi waters expansion
        "2023-12-18",  # Red Sea/Bab-el-Mandeb listing after Houthi campaign
        "2024-04-14",  # Israel-Iran direct exchange, Persian Gulf alert upgrade
    ])

    tickers = ["RNR", "EG", "RLI", "AXS", "ACGL", "SPY"]
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    basket_tickers = ["RNR", "EG", "RLI", "AXS", "ACGL"]

    # Check which basket tickers are available
    available = [t for t in basket_tickers if t in ret.columns]
    if len(available) < 3:
        return mark_failed(sid, f"insufficient tickers available: {available}")

    spy_r = ret["SPY"]
    hold_days = 90  # calendar days, approximate as ~63 trading days

    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for evt_date in event_dates:
        # Find T+1 trading day after event
        future_idx = ret.index[ret.index > evt_date]
        if len(future_idx) < 2:
            continue
        entry_date = future_idx[0]
        entry_pos = ret.index.get_loc(entry_date)

        # Approx 90 cal days ~ 63 trading days
        exit_pos = min(entry_pos + 63, len(ret) - 1)
        exit_date = ret.index[exit_pos]

        # Equal-weight basket returns
        basket_ret = ret[available].iloc[entry_pos:exit_pos]
        ew_ret = basket_ret.mean(axis=1)  # equal-weight daily returns
        spy_window = spy_r.iloc[entry_pos:exit_pos]

        pnl.iloc[entry_pos:exit_pos] += ew_ret.values[:exit_pos - entry_pos]

        basket_cum = float((1 + ew_ret).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "basket_return": round(basket_cum, 4),
            "spy_return": round(spy_cum, 4),
            "alpha": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Lloyd's JWC Listed-Area Expansion -> Long Reinsurer Basket")
    save_result(sid, m, extra={
        "rule": "Enter long equal-weight {RNR, EG, RLI, AXS, ACGL} at T+1 after Lloyd's JWC bulletin adds/expands Gulf/Hormuz/Bab-el-Mandeb to Hull War listed areas; hold 90 days or until removal bulletin",
        "mechanism": "JWC listing triggers immediate jump in marine war risk premiums; reinsurers reprice forthcoming treaty renewals at higher rates; equity re-rates forward earnings before quarterly results confirm hard market",
        "source": "Lloyd's Market Association JWC bulletins (market-standards.co.uk); 4 known events 2019-2024",
        "n_events": len(events),
        "events": events,
        "available_tickers": available,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}, CAGR={m.get('cagr', 'N/A')}")


if __name__ == "__main__":
    main()
