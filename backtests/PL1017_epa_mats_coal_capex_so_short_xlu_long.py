"""PL1017_epa_mats_coal_capex_so_short_xlu_long — EPA MATS Coal Compliance Capex → Short SO / Long XLU Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1017_epa_mats_coal_capex_so_short_xlu_long"

    # EPA rule events: (name, entry_date) — within 5 trading days of Federal Register date
    # Source: Federal Register; FR 77 9304 (Feb 2012); FR 76 48208 (Aug 2011); FR 89 27970 (Apr 2024)
    events = [
        # CSAPR final rule announced 2011-08-03; effective start of cross-state pollution compliance
        {"name": "CSAPR_Aug2011", "entry_date": "2011-08-08"},
        # MATS original rule signed 2011-12-21; published Feb 2012 (effective)
        {"name": "MATS_Dec2011", "entry_date": "2011-12-22"},
        # MATS update tightened Hg/HCl limits effective Apr 25, 2024
        {"name": "MATS_Apr2024", "entry_date": "2024-04-30"},
    ]

    hold_days = 180  # calendar days
    pair_stop = 0.12  # stop if SO outperforms XLU by >12%

    try:
        px = load_prices(["SO", "XLU", "AEP", "NEE", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for col in ["SO", "XLU", "SPY"]:
        if col not in px.columns:
            return mark_failed(sid, f"{col} not in price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    so_r = ret["SO"]
    xlu_r = ret["XLU"]

    # Pair return = (Short SO) + (Long XLU) = XLU_r - SO_r
    pair_r = xlu_r - so_r

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ev in events:
        entry_date = pd.Timestamp(ev["entry_date"])

        # Find closest trading day at or after entry_date
        future_idx = pair_r.index[pair_r.index >= entry_date]
        if len(future_idx) < 5:
            print(f"Skipping {ev['name']}: no data after {entry_date}")
            continue

        entry_idx = future_idx[0]
        entry_loc = pair_r.index.get_loc(entry_idx)

        # Determine exit: either 180 calendar days or pair stop
        exit_date = entry_idx + pd.Timedelta(days=hold_days)
        future_after_entry = pair_r.index[pair_r.index > entry_idx]

        # Build position day by day with stop check
        trade_dates = pair_r.index[(pair_r.index > entry_idx) & (pair_r.index <= exit_date)]
        exit_reason = "time"
        active_dates = []

        # Cumulative SO vs XLU (check pair stop = SO outperforms XLU by >12%)
        so_cum = 1.0
        xlu_cum = 1.0
        for d in trade_dates:
            so_cum *= (1 + so_r.get(d, 0.0))
            xlu_cum *= (1 + xlu_r.get(d, 0.0))
            active_dates.append(d)
            # Stop: SO outperforms XLU by >12% (pair adverse)
            if so_cum - xlu_cum > pair_stop:
                exit_reason = "pair_stop"
                break

        n_days = len(active_dates)
        pair_cum = float((1 + pair_r.loc[active_dates]).prod() - 1) if active_dates else 0.0
        spy_cum = float((1 + spy_r.loc[active_dates]).prod() - 1) if active_dates else 0.0

        event_records.append({
            "name": ev["name"],
            "entry": str(entry_idx.date()),
            "n_days": n_days,
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4),
            "exit_reason": exit_reason,
        })

        for d in active_dates:
            if d in pnl.index:
                pnl[d] += pair_r.get(d, 0.0)

    print(f"Event results ({len(event_records)}):")
    for e in event_records:
        print(f"  {e['name']} ({e['entry']}, {e['n_days']}d, {e['exit_reason']}): pair={e['pair_return']:.2%}, SPY={e['spy_return']:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active PnL days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="EPA MATS Coal -> Short SO / Long XLU Pair")

    save_result(sid, m, extra={
        "rule": "Short SO / Long XLU dollar-neutral pair on EPA coal air-toxics final rule (MATS/CSAPR). Hold 180 calendar days or until pair-stop (SO outperforms XLU >12%).",
        "mechanism": "Southern Company has the largest residual coal fleet among IOU utilities; EPA capex mandates depress earnings vs XLU average. Market re-rates SO lower relative to cleaner utilities on compliance overhang.",
        "source": "SO, XLU via yfinance; EPA Federal Register MATS 2011-12-21 (77 FR 9304), CSAPR 2011-08-03 (76 FR 48208), MATS update 2024-04-25 (89 FR 27970).",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
