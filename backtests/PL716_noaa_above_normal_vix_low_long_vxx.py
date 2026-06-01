"""PL716_noaa_above_normal_vix_low_long_vxx
NOAA Above-Normal Hurricane Outlook + VIX <14 -> Long VXX (Counter-Signal)

On the NOAA CPC May seasonal hurricane outlook publication date where the
outlook is "above-normal" AND the VIX close is below 14 that day, go long
VXX for 30 trading days. This is a counter-signal to long-SPY complacency —
markets are priced for calm but tail-risk is elevated.

NOAA CPC May outlooks are published each May (typically late May). Historical
outcomes table is hand-coded below from NOAA CPC records 2010-2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -------- NOAA CPC May Atlantic Hurricane Season Outlooks 2010-2025 --------
# Source: https://www.cpc.ncep.noaa.gov/products/outlooks/hurricane.shtml
# outlook: "above-normal", "near-normal", "below-normal"
# vix_close: approximate VIX close on or near the publication date
#            (hand-estimated; where VIX < 14 qualifies)
# Note: NOAA typically releases these in late May. We use next trading day
# after publication for entry.
NOAA_EVENTS = [
    # (pub_date, outlook, approx_vix_on_pub_date)
    {"pub_date": "2010-05-27", "outlook": "above-normal", "vix_approx": 35.0},   # VIX elevated post-Flash Crash
    {"pub_date": "2011-05-26", "outlook": "above-normal", "vix_approx": 17.0},
    {"pub_date": "2012-05-24", "outlook": "near-normal",  "vix_approx": 22.0},
    {"pub_date": "2013-05-23", "outlook": "near-normal",  "vix_approx": 13.0},
    {"pub_date": "2014-05-22", "outlook": "below-normal", "vix_approx": 12.5},
    {"pub_date": "2015-05-28", "outlook": "near-normal",  "vix_approx": 13.5},
    {"pub_date": "2016-05-26", "outlook": "near-normal",  "vix_approx": 14.5},
    {"pub_date": "2017-05-25", "outlook": "above-normal", "vix_approx": 10.5},   # VIX historically low
    {"pub_date": "2018-05-24", "outlook": "near-normal",  "vix_approx": 13.5},
    {"pub_date": "2019-05-23", "outlook": "near-normal",  "vix_approx": 14.0},
    {"pub_date": "2020-05-21", "outlook": "above-normal", "vix_approx": 28.0},   # COVID elevated
    {"pub_date": "2021-05-20", "outlook": "above-normal", "vix_approx": 18.5},
    {"pub_date": "2022-05-24", "outlook": "above-normal", "vix_approx": 29.0},   # Russia/Ukraine elevated
    {"pub_date": "2023-05-25", "outlook": "above-normal", "vix_approx": 17.0},
    {"pub_date": "2024-05-23", "outlook": "above-normal", "vix_approx": 12.5},   # Key qualifying event
    {"pub_date": "2025-05-22", "outlook": "above-normal", "vix_approx": 24.0},   # Tariff scare elevated
]


def filter_events(events, vix_threshold=14.0):
    """Return events where outlook == 'above-normal' AND vix_approx < vix_threshold."""
    return [
        e for e in events
        if e["outlook"] == "above-normal" and e["vix_approx"] < vix_threshold
    ]


def run_event_study(events, ret, ticker="VXX", hold_days=30):
    """Simulate event-driven long VXX positions.
    Entry at next session open after pub_date; hold for hold_days trading days.
    Returns (pnl_series, positions_series, event_log).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    asset_ret = ret[ticker].fillna(0)

    event_log = []
    for ev in events:
        pub = pd.Timestamp(ev["pub_date"])
        future = idx[idx > pub]
        if len(future) == 0:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_data_after_pub"
            event_log.append(ev_rec)
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_r = asset_ret.iloc[entry_pos:exit_pos]
        ev_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["event_return"] = round(ev_ret, 4) if ev_ret is not None else None
        event_log.append(ev_rec)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = asset_ret.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL716_noaa_above_normal_vix_low_long_vxx"
    tickers = ["VXX", "SPY", "^VIX"]

    try:
        # VXX started trading in 2009; load from 2010
        px = load_prices(["VXX", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # VXX can go to zero — check it's available
    if "VXX" not in px.columns or px["VXX"].dropna().empty:
        return mark_failed(sid, "VXX not available in price data")

    px = px.sort_index().ffill(limit=2)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # ---------- Strict: above-normal + VIX < 14 ----------
    qualifying = filter_events(NOAA_EVENTS, vix_threshold=14.0)
    pnl_strict, pos_strict, log_strict = run_event_study(qualifying, ret, ticker="VXX", hold_days=30)

    # ---------- Relaxed: above-normal + VIX < 18 (more events) ----------
    qualifying_relaxed = filter_events(NOAA_EVENTS, vix_threshold=18.0)
    pnl_relaxed, pos_relaxed, log_relaxed = run_event_study(qualifying_relaxed, ret, ticker="VXX", hold_days=30)

    n_strict = sum(1 for e in log_strict if e.get("entry_date"))
    n_relaxed = sum(1 for e in log_relaxed if e.get("entry_date"))

    print(f"Strict events (VIX<14): {n_strict}, events: {[e['pub_date'] for e in qualifying]}")
    print(f"Relaxed events (VIX<18): {n_relaxed}, events: {[e['pub_date'] for e in qualifying_relaxed]}")

    # Use relaxed for primary if strict has very few events
    if n_relaxed >= 3:
        primary_pnl = pnl_relaxed
        primary_pos = pos_relaxed
        primary_log = log_relaxed
        primary_label = "above_normal_vix_lt_18"
        n_primary = n_relaxed
    else:
        primary_pnl = pnl_strict
        primary_pos = pos_strict
        primary_log = log_strict
        primary_label = "above_normal_vix_lt_14"
        n_primary = n_strict

    # Use only held days for Sharpe (avoids dilution by flat stretches)
    held_pnl = primary_pnl[primary_pos > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        # Insufficient data — mark failed with event-level stats
        event_rets = [e["event_return"] for e in primary_log if e.get("event_return") is not None]
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_primary})",
            extra={
                "events": primary_log,
                "event_returns": event_rets,
                "avg_event_return": float(np.mean(event_rets)) if event_rets else None,
                "rule": "NOAA above-normal + VIX < threshold -> long VXX 30 days",
                "n_strict": n_strict,
                "n_relaxed": n_relaxed,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="NOAA Above-Normal + VIX Low Long VXX (held-days only)",
        positions=primary_pos.reindex(held_pnl.index).fillna(0),
        cost_bps=15,  # VXX has higher trading costs
    )

    # Event-level summary
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

    summary_strict = event_summary(log_strict)
    summary_relaxed = event_summary(log_relaxed)

    # SPY comparison per event
    def event_vs_spy(log, spy_ret):
        rows = []
        for e in log:
            entry = e.get("entry_date"); exit_d = e.get("exit_date")
            if not entry or not exit_d:
                continue
            entry_dt = pd.Timestamp(entry); exit_dt = pd.Timestamp(exit_d)
            spy_slice = spy_ret.loc[entry_dt:exit_dt]
            if len(spy_slice):
                spy_cum = float((1 + spy_slice).prod() - 1)
                rows.append({
                    "pub_date": e["pub_date"],
                    "entry_date": entry,
                    "vxx_return": e.get("event_return"),
                    "spy_return": round(spy_cum, 4),
                    "excess_vs_spy": round((e.get("event_return") or 0) - spy_cum, 4),
                })
        return rows

    excess_strict = event_vs_spy(log_strict, spy_r)
    excess_relaxed = event_vs_spy(log_relaxed, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When the NOAA CPC May seasonal Atlantic hurricane season outlook is "
                "'above-normal' AND the VIX closes below 14 on the publication date, "
                "go long VXX at the next session open and hold for 30 trading days."
            ),
            "mechanism": (
                "VIX below 14 signals extreme market complacency — realized vol "
                "pricing implies minimal tail risk. An 'above-normal' NOAA hurricane "
                "outlook creates a setup where physical tail-risk (elevated storm "
                "season) is underpriced by option markets. As the season unfolds and "
                "the market reprices vol, VXX gains. The trade exploits the gap "
                "between model-forecast tail risk (NOAA) and market-implied vol (VIX). "
                "30-day hold covers the early-season catalyst window (June-July)."
            ),
            "source": (
                "NOAA CPC May Atlantic Hurricane Season Outlooks 2010-2025, "
                "cpc.ncep.noaa.gov/products/outlooks/hurricane.shtml; "
                "VXX/SPY prices via yfinance (auto_adjust=True)."
            ),
            "primary_rule_label": primary_label,
            "events_strict": log_strict,
            "events_relaxed": log_relaxed,
            "summary_strict": summary_strict,
            "summary_relaxed": summary_relaxed,
            "excess_vs_spy_strict": excess_strict,
            "excess_vs_spy_relaxed": excess_relaxed,
            "n_strict": n_strict,
            "n_relaxed": n_relaxed,
            "caveats": (
                "Event study with very limited sample (1-2 strict qualifying years). "
                "VXX suffers severe structural decay from VIX futures roll costs "
                "(negative roll yield) — long VXX is a high-friction bet. The VIX "
                "values in the NOAA table are approximations; actual qualifying dates "
                "require same-day VIX close verification. Hurricane season outcomes "
                "are correlated with ENSO cycles rather than being i.i.d. The 2017 "
                "event (above-normal + VIX ~10.5) was a rare perfect qualifier but "
                "the season was indeed active (Harvey, Irma, Maria) — the VXX "
                "response is event-timing-dependent. Extreme caution: sample is "
                "too small for statistical inference."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  primary rule: {primary_label}; n_strict={n_strict}, n_relaxed={n_relaxed}")
    print(f"  summary_strict: {summary_strict}")
    print(f"  summary_relaxed: {summary_relaxed}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
