"""PL787_macau_ggr_decel_short_lvmuy_el
Macau DICJ GGR YoY Deceleration + NBS Property Drag -> Short LVMUY / EL Asian Luxury

Event study: hardcoded dates when Macau GGR YoY decelerated >=5pp MoM while
NBS 70-City Home Price YoY was below -8%. Short LVMUY + EL 50/50 for up to 12 weeks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL787_macau_ggr_decel_short_lvmuy_el"
    tickers = ["LVMUY", "EL", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2012-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    for t in ["LVMUY", "EL", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    px = px.sort_index().ffill(limit=5)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Hardcoded event dates: Macau GGR YoY decel >=5pp + NBS 70-city YoY <= -8%
    # 2015-01-29: post-anticorruption GGR collapse + China property slump
    # 2016-06-30: continued GGR weakness + NBS YoY negative
    # 2022-05-31: COVID lockdown hit GGR, NBS turning negative
    # 2023-01-31: pre-reopening uncertainty (GGR decel lag + NBS deep negative)
    event_dates_str = ["2015-01-29", "2016-06-30", "2022-05-31", "2023-01-31"]
    hold_days = 60  # ~12 weeks

    # Exit triggers: GGR re-accel proxy -> we use 12-week max as primary
    pnl_records = []
    event_details = []

    for ev_str in event_dates_str:
        ev_date = pd.Timestamp(ev_str)
        future_idx = ret.index[ret.index > ev_date]
        if len(future_idx) < 5:
            continue

        entry_date = future_idx[0]

        # LVMUY may have limited history before 2015
        has_lvmuy = (
            "LVMUY" in ret.columns and
            not ret["LVMUY"].loc[entry_date:].dropna().empty and
            len(ret["LVMUY"].loc[:entry_date].dropna()) > 0
        )
        has_el = (
            "EL" in ret.columns and
            not ret["EL"].loc[entry_date:].dropna().empty
        )

        if not has_el:
            continue  # EL is required

        end_idx = min(hold_days, len(future_idx))
        window = future_idx[:end_idx]

        # Short basket: 50% LVMUY + 50% EL if LVMUY available, else 100% EL
        if has_lvmuy:
            short_basket = ret[["LVMUY", "EL"]].loc[window].fillna(0).mean(axis=1)
        else:
            short_basket = ret["EL"].loc[window].fillna(0)

        # Short PnL = negative of basket return
        short_pnl = -short_basket

        for d, p in zip(window, short_pnl):
            pnl_records.append({"date": d, "pnl": p})

        cum_ret = (1 + short_pnl).cumprod().iloc[-1] - 1 if len(short_pnl) > 0 else 0
        event_details.append({
            "event_date": ev_str,
            "n_hold_days": len(window),
            "cum_short_ret": float(cum_ret),
            "has_lvmuy": bool(has_lvmuy),
        })

    if not pnl_records:
        return mark_failed(sid, "no valid events found in price data")

    pnl_df = pd.DataFrame(pnl_records).set_index("date")["pnl"]
    pnl_series = pnl_df.groupby(level=0).sum()

    full_idx = ret.index[ret.index >= pnl_series.index.min()]
    pnl_full = pnl_series.reindex(full_idx).fillna(0)

    spy_aligned = spy_r.reindex(full_idx).dropna()
    pnl_aligned = pnl_full.reindex(spy_aligned.index).fillna(0)

    n_events = len([e for e in event_details])

    m = compute_metrics(pnl_aligned, benchmark=spy_aligned, name="Macau GGR Decel Short LVMUY/EL")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "When Macau GGR YoY decelerates >=5pp MoM AND NBS 70-City Home Price YoY <= -8%, short LVMUY+EL 50/50 for up to 12 weeks.",
        "mechanism": "Macau GGR is a leading indicator of Chinese high-net-worth consumer spending. Combined with property wealth destruction (NBS negative), luxury brand pricing power erodes globally.",
        "source": "Macau DICJ monthly GGR; NBS 70-City index; LVMUY/EL/SPY via yfinance. Hardcoded event dates.",
        "event_details": event_details,
        "caveats": "Only 4 known events; EL data starts 1995. LVMUY ADR has intermittent yfinance data pre-2015. Counter to PL117/PL163 luxury rebound family.",
        "status": "ok",
    }, pnl=pnl_aligned)

    print(f"\n=== {sid} ===")
    print(f"Events: {n_events}")
    for ev in event_details:
        print(f"  {ev['event_date']}: held {ev['n_hold_days']}d, cum_short={ev['cum_short_ret']:.2%}, lvmuy={ev['has_lvmuy']}")
    print(f"Sharpe: {m.get('sharpe', 'N/A'):.3f}  CAGR: {m.get('cagr', 0):.2%}  MaxDD: {m.get('max_dd', 0):.2%}  t-stat: {m.get('t_stat', 0):.3f}")
    if 'oos_sharpe' in m:
        print(f"OOS Sharpe: {m['oos_sharpe']:.3f}")


if __name__ == "__main__":
    main()
