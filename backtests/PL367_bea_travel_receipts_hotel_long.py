"""PL367_bea_travel_receipts_hotel_long — BEA International Travel Receipts Surge -> Hotel REITs Long
Using FRED series for travel service exports (BOPTEXP or B0230RC1Q027SBEA).
When quarterly YoY growth > 15%, long HST+PK+MAR for 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL367_bea_travel_receipts_hotel_long"
    try:
        px = load_prices(["HST", "PK", "MAR", "SPY"], start="1999-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load BEA travel receipts from FRED
    travel = None
    for series in ["B0230RC1Q027SBEA", "BOPTEXP", "IEABC"]:
        try:
            travel = load_fred(series, start="1999-01-01").squeeze()
            if travel.dropna().empty:
                travel = None
                continue
            break
        except Exception:
            continue
    if travel is None:
        return mark_failed(sid, "Could not load FRED travel receipts (B0230RC1Q027SBEA / BOPTEXP / IEABC)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Quarterly data, YoY growth
    travel_q = travel.resample("Q").last().dropna()
    travel_yoy = travel_q.pct_change(4)  # 4-quarter YoY

    # Build basket — PK spun off Jan 2017
    def get_basket(date, ret_df):
        if date >= pd.Timestamp("2017-02-01"):
            tickers = [t for t in ["HST", "PK", "MAR"] if t in ret_df.columns]
        else:
            tickers = [t for t in ["HST", "MAR"] if t in ret_df.columns]
        if not tickers:
            return None
        return ret_df[tickers].mean(axis=1)

    # Find signal quarters: YoY > 15%
    triggers = []
    last_trigger = None
    for dt in travel_yoy.dropna().index:
        yoy = float(travel_yoy.loc[dt])
        if yoy > 0.15:
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No travel receipts YoY > 15% events found")

    events = []
    pnl_parts = []
    hold_days = 30

    for trig_date in triggers:
        basket_r = get_basket(trig_date, ret)
        if basket_r is None:
            continue

        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "travel_yoy": round(yoy, 4),
            "basket_30d_return": round(bask_cum, 4),
            "spy_30d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable travel receipt events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="BEA Travel Receipts Surge -> Long Hotel REITs")

    avg_basket = np.mean([e["basket_30d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_30d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "BEA quarterly international travel receipts YoY > 15% -> long HST+PK+MAR 30 days",
        "mechanism": "Surging international visitor spending benefits gateway city hotel REITs through RevPAR acceleration",
        "source": "FRED BEA travel receipts + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_30d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: travel_yoy={e['travel_yoy']*100:.0f}%, basket {e['basket_30d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
