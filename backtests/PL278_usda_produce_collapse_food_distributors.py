"""PL278_usda_produce_collapse_food_distributors
USDA Wholesale Produce Price Collapse Event Study - Long Food Distributors USFD/SYY

On dates of major USDA AMS terminal-market produce-price collapse events
(curated approx. dates around documented multi-category wholesale produce
glut episodes), go long an equal-weight basket of USFD (US Foods) and
SYY (Sysco) at the next session's close after the event, hold 30 trading
days (~6 weeks), and exit at the close.

Hypothesis: distributors buy depressed wholesale produce while restaurant
menu prices remain sticky, expanding gross margins over the next quarter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated event dates from queue's known_events list (approx Mondays in
# periods when USDA AMS terminal market reports documented multi-category
# wholesale produce gluts: post-summer harvest peaks, COVID restaurant
# demand collapse). USFD IPO'd 2016-05, so all event dates are post-IPO.
KNOWN_EVENTS = [
    "2017-10-02",
    "2018-07-09",
    "2019-08-12",
    "2020-04-13",
    "2020-05-04",
    "2021-07-19",
    "2022-09-12",
    "2023-08-14",
    "2024-08-12",
    "2025-08-11",
]


def run_event_study(events, ret, basket, hold_days=30):
    """For each event date, enter long equal-weight basket at next trading
    session's close, hold `hold_days` trading days, exit at close. Returns
    (pnl_series, positions_series, event_log).

    Position is applied to the next day's return (no look-ahead): we assume
    entry at close of session t (the first trading session strictly after
    event date), and capture returns from session t+1 through session
    t+hold_days inclusive.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    basket_ret = ret[basket].fillna(0).mean(axis=1)

    event_log = []
    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue
        entry_close_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_close_dt)
        # Returns realized from entry_pos+1 (first full day held) through
        # entry_pos+hold_days inclusive.
        start_pos = entry_pos + 1
        end_pos = min(entry_pos + 1 + hold_days, len(idx))

        slice_r = basket_ret.iloc[start_pos:end_pos]
        ev_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        record = {
            "event_date": ev_date,
            "entry_close_date": str(entry_close_dt.date()),
            "first_held_date": str(idx[start_pos].date()) if start_pos < len(idx) else None,
            "exit_date": str(idx[end_pos - 1].date()) if end_pos > start_pos else None,
            "n_hold_days": int(end_pos - start_pos),
            "event_return": round(ev_ret, 4) if ev_ret is not None else None,
        }
        event_log.append(record)

        for j in range(start_pos, end_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = basket_ret.iloc[j]

    return pnl, positions, event_log


def event_vs_spy(log, spy_ret):
    rows = []
    for e in log:
        entry = e.get("first_held_date")
        exit_d = e.get("exit_date")
        if not entry or not exit_d:
            continue
        entry_dt = pd.Timestamp(entry)
        exit_dt = pd.Timestamp(exit_d)
        spy_slice = spy_ret.loc[entry_dt:exit_dt]
        if len(spy_slice):
            spy_cum = float((1 + spy_slice).prod() - 1)
            rows.append({
                "event_date": e["event_date"],
                "first_held_date": entry,
                "basket_return": e.get("event_return"),
                "spy_return": round(spy_cum, 4),
                "excess": round((e.get("event_return") or 0) - spy_cum, 4),
            })
    return rows


def event_summary(log):
    rets = [e["event_return"] for e in log if e.get("event_return") is not None]
    if not rets:
        return None
    return {
        "n_events": len(rets),
        "avg_event_return": round(float(np.mean(rets)), 4),
        "median_event_return": round(float(np.median(rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
        "best": round(float(np.max(rets)), 4),
        "worst": round(float(np.min(rets)), 4),
    }


def main():
    sid = "PL278_usda_produce_collapse_food_distributors"
    basket = ["USFD", "SYY"]
    tickers = basket + ["SPY"]

    try:
        # USFD IPO'd 2016-05; start a bit before to have buffer
        px = load_prices(tickers, start="2016-05-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, pos, log = run_event_study(KNOWN_EVENTS, ret, basket, hold_days=30)

    n_events_with_entry = sum(1 for e in log if e.get("first_held_date"))
    held_pnl = pnl[pos > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    summary = event_summary(log)
    excess_log = event_vs_spy(log, spy_r)

    if len(held_pnl) < 30:
        # Not enough held days for metrics
        return mark_failed(
            sid,
            f"insufficient held days (events_with_entry={n_events_with_entry}, held_days={len(held_pnl)})",
            extra={
                "events_log": log,
                "summary": summary,
                "excess_vs_spy": excess_log,
                "n_events": n_events_with_entry,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="USDA Produce Collapse Long USFD+SYY Event Study (held-days)",
        positions=pos.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Annotate compute_metrics output with event count for winner_gate sample-size
    m["n_events"] = n_events_with_entry

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On dates of major USDA AMS terminal-market wholesale-produce "
                "price-collapse events (multi-category gluts), enter long an "
                "equal-weight basket of USFD + SYY at the next session's close, "
                "hold 30 trading days (~6 weeks), exit at the close."
            ),
            "mechanism": (
                "Wholesale produce is a raw-input cost for foodservice "
                "distributors. When USDA AMS terminal-market prices collapse "
                "(post-harvest gluts, COVID restaurant-demand collapse), USFD "
                "and SYY's COGS falls while restaurant menu prices remain "
                "sticky for 1-2 quarters. The compressed input-cost / sticky "
                "output-price spread expands gross margin, which feeds into "
                "earnings revisions over the following quarter and re-rates "
                "the distributors."
            ),
            "source": (
                "USDA Agricultural Marketing Service (AMS) terminal market "
                "news; event dates curated from documented multi-category "
                "wholesale-produce glut periods (post-summer harvest peaks, "
                "COVID demand collapse). Prices via yfinance auto_adjust=True."
            ),
            "tickers": basket,
            "known_events": KNOWN_EVENTS,
            "events_log": log,
            "summary": summary,
            "excess_vs_spy": excess_log,
            "n_events": n_events_with_entry,
            "hold_days": 30,
            "caveats": (
                "Event dates are CURATED approximate Mondays in known glut "
                "periods, not algorithmically derived from raw USDA AMS daily "
                "data. This introduces look-back bias risk: the same dates "
                "could have been chosen for ex-post returns rather than "
                "ex-ante glut signals. Sample size is small (~10 events). "
                "USFD IPO'd 2016-05, limiting history. Distributors' margins "
                "are also exposed to fuel, wages, customer mix (restaurants "
                "vs healthcare/education); produce input is one factor among "
                "many. COVID-window events (2020-04, 2020-05) are confounded "
                "by the broader recovery in foodservice equities."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events_with_entry={n_events_with_entry}, held_days={len(held_pnl)}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
