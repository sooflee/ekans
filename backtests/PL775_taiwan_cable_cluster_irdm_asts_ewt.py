"""PL775_taiwan_cable_cluster_irdm_asts_ewt
Taiwan Strait Subsea Cable Severance Cluster -> Long IRDM/ASTS, Short EWT

Event study: When 2+ Taiwan-region submarine cable cuts occur within 14 days,
go long IRDM (Iridium satellite) + ASTS (AST SpaceMobile) and short EWT (Taiwan ETF).
Hard-coded known cluster events: 2023-02-02 (Matsu #2/#3), 2024-12-01.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL775_taiwan_cable_cluster_irdm_asts_ewt"
    tickers = ["IRDM", "ASTS", "EWT", "SPY"]

    try:
        # ASTS IPO Jan 2022; get data from 2020 for IRDM/EWT history
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2020-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=3)

    # Ensure needed tickers exist
    for t in ["IRDM", "EWT", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Known cable-cluster event dates (T = event confirmation day)
    # Enter on T+1 open (position applied to T+1 daily return)
    event_dates_str = ["2023-02-02", "2024-12-01"]
    hold_days = 20  # use 20-trading-day window as primary

    pnl_records = []
    event_details = []

    for ev_str in event_dates_str:
        ev_date = pd.Timestamp(ev_str)

        # Find T+1 index position
        future_idx = ret.index[ret.index > ev_date]
        if len(future_idx) < 2:
            continue

        # Determine if ASTS data available at entry
        entry_date = future_idx[0]
        has_asts = ("ASTS" in ret.columns and
                    not ret["ASTS"].loc[entry_date:].dropna().empty and
                    not ret["ASTS"].loc[:entry_date].dropna().empty)

        # Build long-side (IRDM + ASTS if available, else IRDM only)
        end_idx = min(hold_days, len(future_idx))
        window = future_idx[:end_idx]

        if has_asts and "ASTS" in ret.columns:
            long_side = ret[["IRDM", "ASTS"]].loc[window].fillna(0).mean(axis=1)
        else:
            long_side = ret["IRDM"].loc[window].fillna(0)

        short_side = ret["EWT"].loc[window].fillna(0)

        # Pair PnL: long IRDM+ASTS basket - short EWT (dollar-neutral as defined)
        # Rule: $0.50 IRDM + $0.50 ASTS long vs $1.00 EWT short
        pair_pnl = long_side - short_side

        for d, p in zip(window, pair_pnl):
            pnl_records.append({"date": d, "pnl": p, "event": ev_str})

        cum = (1 + pair_pnl).cumprod().iloc[-1] - 1 if len(pair_pnl) > 0 else 0
        entry_long = (1 + long_side).cumprod().iloc[-1] - 1 if len(long_side) > 0 else 0
        entry_short = (1 + short_side).cumprod().iloc[-1] - 1 if len(short_side) > 0 else 0
        event_details.append({
            "event_date": ev_str,
            "long_basket_ret": float(entry_long),
            "short_ewt_ret": float(entry_short),
            "pair_total_ret": float(cum),
            "has_asts": bool(has_asts),
        })

    if not pnl_records:
        return mark_failed(sid, "no valid events found in price data")

    # Build daily pnl series (non-event days = 0)
    pnl_df = pd.DataFrame(pnl_records).set_index("date")["pnl"]
    # Aggregate if multiple events overlap
    pnl_series = pnl_df.groupby(level=0).sum()

    # Reindex to full trading calendar and fill with 0
    full_idx = ret.index[ret.index >= pnl_series.index.min()]
    pnl_full = pnl_series.reindex(full_idx).fillna(0)

    spy_aligned = spy_r.reindex(full_idx).dropna()
    pnl_aligned = pnl_full.reindex(spy_aligned.index).fillna(0)

    m = compute_metrics(pnl_aligned, benchmark=spy_aligned, name="Taiwan Cable Cluster Long IRDM/ASTS Short EWT")
    m["n_events"] = len(event_dates_str)

    save_result(sid, m, extra={
        "rule": "On 2+ Taiwan-region submarine cable cuts within 14 days, enter long IRDM+ASTS equal-weight, short EWT. Exit after 20 trading days or RFS.",
        "mechanism": "Satellite comms (IRDM, ASTS) benefit from terrestrial/subsea outage demand spike; EWT suffers geopolitical risk premium and economic isolation fears.",
        "source": "TeleGeography Submarine Cable Map; Taiwan MODA press releases; hardcoded event dates 2023-02-02 and 2024-12-01",
        "event_details": event_details,
        "caveats": "Only 2 known cluster events; n_events too small for robust statistical inference. ASTS not available pre-2022.",
        "status": "ok",
    }, pnl=pnl_aligned)

    print(f"\n=== {sid} ===")
    print(f"Events: {len(event_dates_str)}, n_days active: {int((pnl_full != 0).sum())}")
    for ev in event_details:
        print(f"  {ev['event_date']}: long={ev['long_basket_ret']:.2%}, short_ewt={ev['short_ewt_ret']:.2%}, pair={ev['pair_total_ret']:.2%}, asts={ev['has_asts']}")
    print(f"Sharpe: {m.get('sharpe', 'N/A'):.3f}  CAGR: {m.get('cagr', 0):.2%}  MaxDD: {m.get('max_dd', 0):.2%}  t-stat: {m.get('t_stat', 0):.3f}")


if __name__ == "__main__":
    main()
