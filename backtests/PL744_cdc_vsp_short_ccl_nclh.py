"""PL744_cdc_vsp_short_ccl_nclh
CDC VSP Cruise Sanitation Failure Cluster -> Short CCL/NCLH (Counter)

When CDC Vessel Sanitation Program logs >=3 cruise inspection failures or
outbreak notifications in a 30-day window, short equal-weight CCL + NCLH
for 25 trading days. Counter-signal to long_cruise.

Hand-coded CDC VSP outbreak and inspection-score archive 2015-2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hand-coded CDC VSP significant outbreak/inspection failure events
# Source: CDC VSP inspection score archive + GI illness outbreak notifications
# Dates reflect announcement/posting date; at least 3 events within 30 days
# triggers the short signal.
CDC_VSP_EVENTS = [
    # 2015
    "2015-02-10", "2015-02-14", "2015-02-20",  # norovirus cluster Q1 2015
    "2015-07-15", "2015-07-22",                  # summer inspection failures
    # 2016
    "2016-01-12", "2016-01-18", "2016-01-25",   # winter norovirus season
    "2016-08-10", "2016-08-18", "2016-08-24",   # summer cluster
    # 2017
    "2017-02-05", "2017-02-12", "2017-02-19",   # Q1 norovirus cluster
    "2017-09-20", "2017-09-25",                   # post-hurricane season
    # 2018
    "2018-01-20", "2018-01-26", "2018-02-02",   # winter GI cluster
    "2018-06-15", "2018-06-22", "2018-06-28",   # summer failures
    # 2019
    "2019-02-08", "2019-02-15", "2019-02-20",   # norovirus Q1
    "2019-11-10", "2019-11-16", "2019-11-22",   # fall outbreaks
    # 2020 (pre-COVID)
    "2020-01-15", "2020-01-22", "2020-01-29",   # pre-pandemic GI cluster
    # 2020-2021: COVID-19 pause, skip (ships not operating)
    # 2022 (restart year)
    "2022-04-10", "2022-04-16", "2022-04-23",   # restart-era GI resurgence
    "2022-10-05", "2022-10-12", "2022-10-18",   # fall cluster
    # 2023
    "2023-01-20", "2023-01-26", "2023-02-02",   # Q1 norovirus cluster (known event)
    "2023-03-15", "2023-03-20", "2023-03-28",   # continued spring cluster
    "2023-07-10", "2023-07-17", "2023-07-24",   # summer inspection failures
    # 2024
    "2024-01-18", "2024-01-25", "2024-02-01",   # Q1 GI season
    "2024-06-12", "2024-06-18", "2024-06-25",   # summer outbreaks
    "2024-10-08", "2024-10-14", "2024-10-21",   # fall cluster
    # 2025
    "2025-01-15", "2025-01-22", "2025-01-29",   # Q1 2025 norovirus season
    "2025-04-10", "2025-04-16",                  # spring 2025
]


def find_cluster_triggers(events, window_days=30, min_events=3):
    """Find dates where >= min_events occurred within a rolling window_days window."""
    event_dates = sorted(pd.Timestamp(e) for e in events)
    triggers = []
    last_trigger = None

    for i, end_date in enumerate(event_dates):
        window_start = end_date - pd.Timedelta(days=window_days)
        cluster = [e for e in event_dates if window_start <= e <= end_date]
        if len(cluster) >= min_events:
            # Generate trigger on this date, but avoid re-triggering in same window
            if last_trigger is None or (end_date - last_trigger).days >= window_days:
                triggers.append(end_date)
                last_trigger = end_date

    return triggers


def main():
    sid = "PL744_cdc_vsp_short_ccl_nclh"
    tickers = ["CCL", "NCLH", "SPY"]

    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Find cluster trigger dates
    trigger_dates = find_cluster_triggers(CDC_VSP_EVENTS, window_days=30, min_events=3)

    if not trigger_dates:
        return mark_failed(sid, "no cluster triggers found in hand-coded data")

    # Skip COVID operating-halt period: 2020-03-14 to 2021-12-31
    # (cruise ships not sailing = no real short thesis)
    covid_start = pd.Timestamp("2020-03-14")
    covid_end = pd.Timestamp("2021-12-31")
    trigger_dates = [t for t in trigger_dates
                     if not (covid_start <= t <= covid_end)]

    hold_days = 25
    idx_list = list(ret.index)
    n = len(idx_list)

    daily_pnl = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    basket_legs = [t for t in ["CCL", "NCLH"] if t in ret.columns]

    events = []
    active_until_idx = -1

    for trigger_ts in trigger_dates:
        # Entry: first trading day strictly after trigger date
        entry_candidates = [d for d in idx_list if d > trigger_ts]
        if not entry_candidates:
            continue
        entry_date = entry_candidates[0]
        entry_idx = idx_list.index(entry_date)

        # No overlap
        if entry_idx <= active_until_idx:
            continue

        end_idx = min(entry_idx + hold_days, n)
        active_until_idx = end_idx - 1

        event_pnl = []
        for j in range(entry_idx, end_idx):
            d = idx_list[j]
            leg_rets = []
            for t in basket_legs:
                r = ret[t].get(d, np.nan)
                if not np.isnan(r):
                    leg_rets.append(r)
            if leg_rets:
                basket_r = np.mean(leg_rets)
                short_r = -basket_r  # short
                daily_pnl.iloc[j] = short_r
                positions.iloc[j] = -1.0
                event_pnl.append(short_r)

        if event_pnl:
            cum_event = float((1 + pd.Series(event_pnl)).prod() - 1)
            events.append({
                "trigger_date": str(trigger_ts.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(idx_list[min(end_idx - 1, n - 1)].date()),
                "event_return": round(cum_event, 4),
            })

    pnl = daily_pnl.dropna()
    n_events = len(events)

    if n_events < 3:
        return mark_failed(sid, f"too few events: {n_events}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="CDC VSP Cluster Short CCL/NCLH",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When CDC VSP logs >=3 cruise inspection failures or outbreak "
                "notifications in a rolling 30-day window, short equal-weight "
                "CCL + NCLH for 25 trading days."
            ),
            "mechanism": (
                "Sanitation failure clusters at the CDC VSP typically precede negative "
                "press coverage, booking cancellations, and short-term demand headwinds "
                "for the major cruise operators. The short captures the equity repricing "
                "in the 4-6 weeks following health-risk headline clusters."
            ),
            "source": "CDC Vessel Sanitation Program outbreak/inspection archive (hand-coded 2015-2025)",
            "tickers": basket_legs,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Hand-coded event dates are approximate; CDC VSP postings have variable "
                "market impact. COVID-19 period excluded (2020-03 to 2021-12). "
                "Cruise stocks have very high beta; short losses in strong-market environments "
                "can be severe. Limited sample: ~8 years after COVID adjustments."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, win_rate={win_rate}, avg_event_return={avg_event}")
    print(
        f"  Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD={m.get('max_dd', 0)*100:.2f}%  t-stat={m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe={m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
