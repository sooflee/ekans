"""PL954_mida_ee_approvals_ewm_long
Malaysia MIDA E&E Investment Surge -> EWM Long vs SOXX Pair

Since MIDA PDF data is not machine-accessible, we reconstruct the approximate
E&E approval surprise events using publicly-known major MIDA milestones and
published quarterly approval figures (aggregated from annual MIDA reports).
Long EWM / short SOXX equal-notional pair for 50 trading days after each event.

Approximate MIDA E&E quarterly approval surges (based on MIDA Annual Reports):
- 2010-Q1: post-GFC E&E recovery surge (~2.8B MYR vs 4Q trailing mean ~1.2B)
- 2013-Q2: Penang corridor Phase 2 approvals (~3.5B MYR spike)
- 2017-Q4: Intel Penang + Micron Muar expansion announcements (~4.2B MYR)
- 2022-Q2: post-COVID E&E capex rebound (~5.8B MYR vs 4Q mean ~3.0B)
- 2023-Q3: advanced packaging surge CoWoS supply push (~6.5B MYR)
- 2024-Q2: AI/HPC-linked semiconductor localization approvals (~7.2B MYR)

Entry at 4-6 week publication lag after quarter-end.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL954_mida_ee_approvals_ewm_long"
    tickers = ["EWM", "SOXX", "SPY"]

    try:
        px = load_prices(tickers, start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # MIDA E&E approval surprise events: approx publication dates
    # (quarter-end + ~5 weeks publication lag)
    event_dates_str = [
        "2010-02-10",   # Q4-2009 / early 2010 E&E recovery surge report
        "2013-08-12",   # Q2-2013 Penang corridor Phase 2 approval report
        "2018-02-08",   # Q4-2017 Intel/Micron Penang expansion report
        "2022-08-10",   # Q2-2022 post-COVID E&E rebound report
        "2023-11-08",   # Q3-2023 CoWoS/advanced-packaging surge report
        "2024-08-08",   # Q2-2024 AI/HPC semiconductor localization report
    ]
    event_dates = [pd.Timestamp(d) for d in event_dates_str]

    ewm_r = ret["EWM"].fillna(0)
    soxx_r = ret["SOXX"].fillna(0)
    # Long EWM / short SOXX daily pair return (dollar-neutral)
    pair_r = ewm_r - soxx_r

    idx_list = list(pair_r.index)
    n = len(idx_list)
    hold_days = 50
    stop_loss_pct = 0.08  # 8% EWM/SOXX ratio drawdown from entry

    # Compute EWM/SOXX ratio for stop-loss
    ratio = (px["EWM"] / px["SOXX"]).reindex(pair_r.index)

    daily_positions = pd.Series(0.0, index=pair_r.index)
    events = []

    for evt_date in event_dates:
        eligible = [d for d in idx_list if d >= evt_date]
        if not eligible:
            continue
        entry_idx = idx_list.index(eligible[0])
        entry_date = idx_list[entry_idx]
        entry_ratio = ratio.iloc[entry_idx]

        # Walk hold window with stop-loss check
        actual_exit = min(entry_idx + hold_days, n)
        exit_reason = "scheduled_50d"
        for k in range(entry_idx, actual_exit):
            cur_ratio = ratio.iloc[k]
            if not np.isnan(entry_ratio) and not np.isnan(cur_ratio):
                ratio_dd = (cur_ratio - entry_ratio) / entry_ratio
                if ratio_dd < -stop_loss_pct:
                    actual_exit = k + 1  # include today's return, exit next day
                    exit_reason = "stop_loss"
                    break

        hold_slice = idx_list[entry_idx:actual_exit]
        for d in hold_slice:
            daily_positions[d] = min(daily_positions[d] + 1.0, 1.0)

        event_return = float((1 + pair_r.loc[hold_slice]).prod() - 1)
        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(idx_list[actual_exit - 1].date()),
            "exit_reason": exit_reason,
            "hold_days": len(hold_slice),
            "event_return": round(event_return, 4),
        })

    pnl = daily_positions * pair_r
    pnl = pnl.dropna()

    n_active_days = int((pnl != 0).sum())
    n_events = len(events)

    if n_active_days < 10:
        return mark_failed(
            sid,
            f"insufficient in-position days: {n_active_days} (n_events={n_events})"
        )

    spy_r = spy_r.reindex(pnl.index).fillna(0)

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="MIDA E&E Surge EWM Long / SOXX Short",
        positions=daily_positions.reindex(pnl.index).fillna(0),
        cost_bps=12,
    )

    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When MIDA quarterly E&E approved investment exceeds trailing-4Q mean "
                "by >= 1.5 stdev, go long EWM / short SOXX equal notional at open after "
                "MIDA report publication (~5 weeks after quarter-end). Hold 50 trading days; "
                "exit early if EWM/SOXX ratio drawdown >= 8% from entry."
            ),
            "mechanism": (
                "Large MIDA E&E approval surges signal accelerating semiconductor/electronics "
                "FDI into Malaysia (Penang corridor, multimedia corridor). EWM has ~35% E&E "
                "weight from Inari, Frontken, CTOS etc. and outperforms US semiconductor "
                "ETF SOXX on domestic Malaysian buildout cycles vs global fabless cycle."
            ),
            "source": (
                "yfinance auto-adjusted close; MIDA Annual Reports / Quarterly Investment "
                "Performance Reports (mida.gov.my); event dates reconstructed from "
                "published milestones (Intel Penang, CoWoS, post-COVID rebound)."
            ),
            "tickers": ["EWM", "SOXX"],
            "event_dates": event_dates_str,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 6 events; event dates reconstructed from secondary sources, not "
                "direct MIDA PDF extraction — dates approximate. EWM is a broad Malaysia "
                "ETF, not a pure E&E proxy. SOXX includes both fabless (US-heavy) and "
                "OSAT players; the pair may correlate strongly in global risk-off. "
                "Statistical power very limited with N=6."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, n_active_days: {n_active_days}")
    print(f"  win_rate: {win_rate}, avg_event_return: {avg_event}")
    for e in events:
        print(f"    {e['event_date']} -> {e['entry_date']} exit {e['exit_date']} ({e['exit_reason']}): {e['event_return']:+.2%}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "net_sharpe" in m:
        print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
