"""PL960_idf_tzav8_mobilization_eis_short_gld_long
IDF Tzav 8 Mass Mobilization Surge: Short EIS, Long GLD (Regional War Premium)

Event-study: on each IDF Tzav 8 mass mobilization order, enter short EIS /
long GLD equal-dollar pair for 20 trading days. EIS = iShares MSCI Israel ETF
(conflict discount); GLD = SPDR Gold ETF (safe-haven war premium).

Known Tzav 8 events:
- 2023-10-08: mass Tzav 8 day after Hamas Oct 7 attack (~360,000 reservists)
- 2024-05-06: Rafah ground operation Tzav 8 for S.Gaza/Southern Command
- 2024-09-30: N.Command Tzav 8 for Lebanon ground operation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL960_idf_tzav8_mobilization_eis_short_gld_long"
    tickers = ["EIS", "GLD", "SPY"]

    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Tzav 8 event dates (known confirmed mobilization events)
    event_dates_str = [
        "2023-10-08",   # Day-1 of mass Tzav 8 after Oct 7 Hamas attack
        "2024-05-06",   # Rafah ground operation Tzav 8
        "2024-09-30",   # N.Command Lebanon ground operation Tzav 8
    ]
    event_dates = [pd.Timestamp(d) for d in event_dates_str]

    eis_r = ret["EIS"].fillna(0)
    gld_r = ret["GLD"].fillna(0)
    # Short EIS / Long GLD pair daily return (dollar-neutral)
    pair_r = gld_r - eis_r

    idx_list = list(pair_r.index)
    n = len(idx_list)
    hold_days = 20
    stop_loss_pct_eis = 0.05   # exit if EIS gains >5% from entry (thesis failing)

    # Track EIS price for stop-loss
    eis_px = px["EIS"]

    daily_positions = pd.Series(0.0, index=pair_r.index)
    events = []

    for evt_date in event_dates:
        # T+0 entry (Oct 7 was Sunday; Oct 8 is first trading day)
        eligible = [d for d in idx_list if d >= evt_date]
        if not eligible:
            continue
        entry_idx = idx_list.index(eligible[0])
        entry_date = idx_list[entry_idx]
        entry_eis = eis_px.iloc[entry_idx]

        actual_exit = min(entry_idx + hold_days, n)
        exit_reason = "scheduled_20d"
        for k in range(entry_idx, actual_exit):
            cur_eis = eis_px.iloc[k]
            if not np.isnan(entry_eis) and entry_eis > 0 and not np.isnan(cur_eis):
                eis_chg = (cur_eis - entry_eis) / entry_eis
                if eis_chg > stop_loss_pct_eis:
                    actual_exit = k + 1
                    exit_reason = "stop_loss_eis_gain"
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

    if n_active_days < 5:
        return mark_failed(
            sid,
            f"insufficient in-position days: {n_active_days} (n_events={n_events})"
        )

    spy_r = spy_r.reindex(pnl.index).fillna(0)

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="IDF Tzav8 Short EIS / Long GLD",
        positions=daily_positions.reindex(pnl.index).fillna(0),
        cost_bps=15,
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
                "When IDF confirms Tzav 8 call-up >25,000 reservists or N.Command "
                "divisions, enter short EIS / long GLD equal notional at T+0 open. "
                "Hold 20 trading days; exit early if EIS gains >5% from entry."
            ),
            "mechanism": (
                "Major IDF mobilizations create Israel-specific conflict discount "
                "(EIS drops on IDF call-up news) while simultaneously triggering "
                "gold safe-haven demand (GLD bids on geopolitical risk in Israel/Iran "
                "proxy axis). The pair trade isolates geopolitical-risk premium "
                "while neutralizing broad market moves."
            ),
            "source": (
                "yfinance auto-adjusted close; IDF Spokesperson RSS (idf.il/en); "
                "Reuters/AP coverage of Oct 7 2023, Rafah May 2024, Lebanon Sep 2024."
            ),
            "tickers": ["EIS", "GLD"],
            "event_dates": event_dates_str,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 3 events — extremely small sample; dominated by Oct 2023 "
                "which is the most severe event. EIS is a thin ETF with limited "
                "short availability. Ceasefire and de-escalation announcements "
                "can rapidly reverse pair. Results highly regime-dependent on "
                "Middle East conflict escalation cycle."
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
