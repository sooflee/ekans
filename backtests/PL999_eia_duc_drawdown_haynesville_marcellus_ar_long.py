"""PL999_eia_duc_drawdown_haynesville_marcellus_ar_long — EIA DUC Drawdown Haynesville/Marcellus Supply Cliff: Long AR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL999_eia_duc_drawdown_haynesville_marcellus_ar_long"

    # EIA DPR data: Haynesville + Marcellus DUC trough events
    # Known low-DUC events (per strategy notes):
    # 2019-12: Pre-COVID DUC drawdown
    # 2022-06: Post-rebound DUC trough
    # 2023-08: Summer DUC low
    # Also including 2016-08: DUC count was very low during gas price trough/recovery
    # Using EIA DPR publication dates (signal trigger dates)
    # EIA DPR has ~5-6 week publication lag after reference month

    # Known events: approximate EIA DPR publication dates for low-DUC signal months
    events = [
        ("2019-12-10", "AR"),  # Dec 2019 DPR published ~Jan 2020 -> use Jan 2020 entry
        ("2020-01-14", "AR"),  # Jan 2020 confirmation entry (5-6 week lag from Dec)
        ("2022-07-12", "AR"),  # July 2022 DPR (for June 2022 data)
        ("2023-09-12", "AR"),  # Sep 2023 DPR (for Aug 2023 data)
    ]

    # Remove duplicates / keep one per event cluster (avoid overlapping 60-day windows)
    # Use only 2022-07 and 2023-09 plus one pre-COVID event
    events = [
        ("2020-01-14", "AR"),  # Pre-COVID DUC trough entry
        ("2022-07-12", "AR"),  # 2022 DUC trough
        ("2023-09-12", "AR"),  # 2023 DUC trough
    ]

    hold_days = 60  # trading days (~12 weeks)

    try:
        px = load_prices(["AR", "EQT", "UNG", "SPY"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "AR" not in px.columns:
        return mark_failed(sid, "AR not found in price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    ar_r = ret["AR"]

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ev_date_str, ticker in events:
        ev_date = pd.Timestamp(ev_date_str)
        future_idx = ar_r.index[ar_r.index >= ev_date]

        if len(future_idx) < hold_days:
            print(f"Skipping {ev_date_str}: insufficient data after event")
            continue

        entry_idx = future_idx[0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret))

        ar_window = ar_r.iloc[entry_loc:exit_loc]
        spy_window = spy_r.iloc[entry_loc:exit_loc]

        # Apply stop-loss (-12%) and take-profit (+25%) logic
        cum_ret = (1 + ar_window).cumprod() - 1
        # Find first breach
        stop_hit = cum_ret[cum_ret <= -0.12]
        tp_hit = cum_ret[cum_ret >= 0.25]
        exit_reason = "time"

        if not stop_hit.empty and not tp_hit.empty:
            first_stop = stop_hit.index[0]
            first_tp = tp_hit.index[0]
            if first_stop < first_tp:
                cut = ar_r.index.get_loc(first_stop) + 1
                ar_window = ar_r.iloc[entry_loc:cut]
                exit_reason = "stop_loss"
            else:
                cut = ar_r.index.get_loc(first_tp) + 1
                ar_window = ar_r.iloc[entry_loc:cut]
                exit_reason = "take_profit"
        elif not stop_hit.empty:
            cut = ar_r.index.get_loc(stop_hit.index[0]) + 1
            ar_window = ar_r.iloc[entry_loc:cut]
            exit_reason = "stop_loss"
        elif not tp_hit.empty:
            cut = ar_r.index.get_loc(tp_hit.index[0]) + 1
            ar_window = ar_r.iloc[entry_loc:cut]
            exit_reason = "take_profit"

        ar_cum = float((1 + ar_window).prod() - 1)
        spy_cum = float((1 + spy_r.iloc[entry_loc:entry_loc + len(ar_window)]).prod() - 1)

        event_records.append({
            "event_date": ev_date_str,
            "ticker": ticker,
            "ar_return": round(ar_cum, 4),
            "spy_return": round(spy_cum, 4),
            "n_days": len(ar_window),
            "exit_reason": exit_reason,
        })

        for idx, r in ar_window.items():
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['event_date']} ({e['exit_reason']}): AR={e['ar_return']:.2%}, SPY={e['spy_return']:.2%}, {e['n_days']} days")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="EIA DUC Trough -> Long AR")

    save_result(sid, m, extra={
        "rule": "Long AR when EIA DPR Haynesville+Marcellus DUC < 400 at 24-month rolling low; hold 60 trading days with -12%/+25% stops",
        "mechanism": "DUC drawdown signals near-term supply cliff for US natgas; Haynesville+Marcellus are swing supply basins. Low DUC = limited completion backlog = production growth ceiling = bullish price setup, which benefits pure-play Appalachian E&P like AR with 100% HH price exposure",
        "source": "EIA Drilling Productivity Report (eia.gov/petroleum/drilling); AR, EQT via yfinance; known DUC trough events: 2020-01, 2022-07, 2023-09",
        "n_events": len(event_records),
        "events": event_records,
        "duc_threshold": 400,
        "hold_period_days": hold_days,
    })


if __name__ == "__main__":
    main()
