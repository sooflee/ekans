"""PL761_nyts_youth_short_tobacco — NYTS Youth Nicotine + FDA Enforcement -> Short MO/PM/BTI"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL761_nyts_youth_short_tobacco"

    # CDC National Youth Tobacco Survey (NYTS) annual results
    # + FDA deeming-rule enforcement actions in same quarter
    # Condition: Youth e-cigarette/nicotine use UP >=2pp YoY AND FDA enforcement same quarter
    #
    # NYTS annual data (release dates and youth e-cig use rates):
    # 2018: 20.8% high school (published ~Nov 2018); 2017 baseline = 11.7% → +9.1pp → QUALIFIES
    # 2019: 27.5% HS (published ~Aug 2019); 2018 = 20.8% → +6.7pp → QUALIFIES
    #   FDA took enforcement against JUUL etc in 2018-2019 period
    # 2020: 19.6% HS (published ~Sep 2020); 2019 = 27.5% → -7.9pp → DOES NOT QUALIFY (decline)
    # 2021: 11.3% HS (published ~Sep 2021) → -8.3pp → no
    # 2022: 14.1% HS (published ~Nov 2022) → +2.8pp → QUALIFIES
    #   FDA enforcement actions against unauthorized flavored disposables in Q3/Q4 2022
    # 2023: 10.0% HS (published ~Nov 2023) → -4.1pp → no
    # 2024: est. ~10% → marginal
    #
    # Qualifying events (NYTS release date, YoY change >=+2pp AND FDA enforcement same quarter):
    # Using publication/media date as trigger:
    qualifying_events = [
        # date_str, yoy_pp_change, youth_rate_pct, note
        ("2018-11-15", 9.1, 20.8, "2018 NYTS spike; FDA enforcement vs JUUL Oct 2018"),
        ("2019-08-22", 6.7, 27.5, "2019 NYTS spike; FDA 2019 enforcement actions"),
        ("2022-11-03", 2.8, 14.1, "2022 NYTS increase; FDA enforcement vs disposables Q3 2022"),
    ]

    try:
        px = load_prices(["MO", "PM", "BTI", "SPY"], start="2015-01-01")
        ret = daily_returns(px)
        spy_r = ret["SPY"]
        mo_r = ret["MO"]
        pm_r = ret["PM"]
        bti_r = ret["BTI"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 45  # 45 trading days

    # Equal-weight basket: MO + PM + BTI, all shorted
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for event_date_str, yoy_pp, youth_rate, note in qualifying_events:
        event_date = pd.Timestamp(event_date_str)

        # Find first trading day on or after event date (use SPY as trading calendar)
        mask = spy_r.index >= event_date
        if mask.sum() < hold:
            print(f"  Skip {event_date_str}: insufficient future data")
            continue

        entry_idx = spy_r.index[mask][0]
        sp_idx = spy_r.index.get_loc(entry_idx)
        sp_end = min(sp_idx + hold, len(spy_r))

        # Build equal-weight short basket returns
        basket_returns = {}
        valid_tickers = []
        for tkr, r_series in [("MO", mo_r), ("PM", pm_r), ("BTI", bti_r)]:
            if entry_idx in r_series.index:
                p = r_series.index.get_loc(entry_idx)
                ep = min(p + hold, len(r_series))
                window = r_series.iloc[p:ep]
                if len(window) >= hold // 2:  # at least half the window
                    basket_returns[tkr] = window
                    valid_tickers.append(tkr)

        if not valid_tickers:
            print(f"  Skip {event_date_str}: no valid tickers")
            continue

        # Equal-weight short: average the returns of all tickers, then negate
        min_len = min(len(v) for v in basket_returns.values())
        avg_long = np.mean([basket_returns[t].iloc[:min_len].values for t in valid_tickers], axis=0)
        short_returns = -avg_long

        window_dates = spy_r.index[sp_idx: sp_idx + min_len]
        pnl.loc[window_dates] = short_returns[:len(window_dates)]

        cum_basket = float((1 + pd.Series(avg_long)).prod() - 1)
        cum_short = -cum_basket

        spy_window = spy_r.iloc[sp_idx:sp_end]
        cum_spy = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "yoy_pp_change": yoy_pp,
            "youth_rate_pct": youth_rate,
            "note": note,
            "valid_tickers": valid_tickers,
            "short_basket_return": round(cum_short, 4),
            "basket_long_return": round(cum_basket, 4),
            "spy_return": round(cum_spy, 4),
        })
        print(f"  {event_date_str} (+{yoy_pp:.1f}pp): short basket {cum_short*100:.1f}% "
              f"vs SPY {cum_spy*100:.1f}%")

    print(f"Total qualifying events: {len(events)}")

    if len(events) == 0:
        return mark_failed(sid, "no qualifying NYTS + FDA enforcement events found")

    active_pnl = pnl[pnl != 0]

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active PnL days ({len(active_pnl)}): "
                               f"{len(events)} event(s) with {hold}-day windows")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NYTS Youth Nicotine -> Short MO/PM/BTI")
    save_result(sid, m, extra={
        "rule": "Short equal-weight MO+PM+BTI for 45 trading days when CDC NYTS shows "
                "youth e-cig use up >=2pp YoY AND FDA enforcement actions same quarter",
        "mechanism": "Rising youth nicotine use triggers FDA regulatory response; "
                     "enforcement risk compresses tobacco sector multiples and raises "
                     "litigation/regulatory tail risk, weighing on tobacco equities.",
        "source": "CDC NYTS annual reports; FDA enforcement press releases; "
                  "yfinance MO, PM, BTI, SPY",
        "n_events": len(events),
        "events": events,
        "caveats": "Annual frequency limits event count. FDA enforcement timing is "
                   "quarterly-approximate. BTI has FX exposure (GBP/USD) that adds noise. "
                   "bt_feasibility=3 reflects sparse event history.",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
