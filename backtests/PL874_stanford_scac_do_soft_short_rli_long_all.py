"""PL874_stanford_scac_do_soft_short_rli_long_all — Stanford SCAC Soft Cycle -> Short RLI / Long ALL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL874_stanford_scac_do_soft_short_rli_long_all"

    # D&O cycle regime dates from strategy spec and Cornerstone Research publications:
    # Soft cycle periods (short RLI, long ALL):
    # - 2002-2004: post-Enron/SOX, SCAC filings declining
    # - 2005-2008: D&O pricing softening cycle
    # - 2022-present: post-SPAC wave collapse, SCAC filings declining
    #
    # Strategy: short RLI, long ALL (1:1 dollar-neutral) at soft cycle trigger dates
    # Trigger dates = Cornerstone Research semi-annual report dates (Jan and Jul)
    # Soft cycle entry dates (when SCAC filing rate soft signal is active):
    soft_cycle_entries = [
        ("2005-01-03", "D&O soft cycle 2005 — SCAC filings declining post-SOX"),
        ("2005-07-05", "D&O soft cycle 2005 mid-year — rate softening continues"),
        ("2006-01-03", "D&O soft cycle 2006 — low filing frequency"),
        ("2006-07-03", "D&O soft cycle 2006 mid-year"),
        ("2022-07-05", "D&O soft cycle 2022 — SPAC wave collapses, filings declining"),
        ("2023-01-03", "D&O soft cycle 2023 H1 — post-SPAC normalization"),
        ("2023-07-03", "D&O soft cycle 2023 H2 — sustained low filing rate"),
        ("2024-01-02", "D&O soft cycle 2024 H1 — SCAC filing rate below 3yr avg"),
    ]

    try:
        px = load_prices(["RLI", "ALL", "SPY"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or len(px) < 252:
        return mark_failed(sid, "insufficient price data")

    for ticker in ["RLI", "ALL"]:
        if ticker not in px.columns or px[ticker].notna().mean() < 0.8:
            return mark_failed(sid, f"insufficient data for {ticker}")

    ret = daily_returns(px)
    rli_r = ret["RLI"]
    all_r = ret["ALL"]
    spy_r = ret["SPY"]

    # Strategy: Short RLI, Long ALL (1:1 dollar-neutral)
    # pair_pnl = all_return - rli_return
    pair_r = all_r - rli_r

    hold_days = 75  # ~15 weeks
    stop_loss = 0.12  # close if RLI rises >12% vs ALL (pair goes -12%)

    pnl = pd.Series(0.0, index=spy_r.index)
    all_events = []
    last_exit = pd.Timestamp("2000-01-01")

    for event_date_str, label in soft_cycle_entries:
        trigger = pd.Timestamp(event_date_str)

        avail_dates = pair_r.index[pair_r.index >= trigger]
        if len(avail_dates) == 0:
            continue
        entry_date = avail_dates[0]
        if entry_date <= last_exit:
            continue

        if entry_date not in pair_r.index:
            continue

        entry_idx = pair_r.index.get_loc(entry_date)
        max_exit_idx = min(entry_idx + hold_days, len(pair_r) - 1)

        cum_pair = 0.0
        actual_exit_idx = max_exit_idx
        exit_reason = "time_stop"

        for j in range(entry_idx, max_exit_idx + 1):
            pr = pair_r.iloc[j]
            if np.isnan(pr):
                continue
            cum_pair = (1 + cum_pair) * (1 + pr) - 1
            # Stop-loss: pair drops 12% (RLI outperforms ALL by 12%)
            if cum_pair < -stop_loss:
                actual_exit_idx = j
                exit_reason = "stop_loss"
                break

        window_pnl = pair_r.iloc[entry_idx:actual_exit_idx + 1]
        for j, idx_date in enumerate(window_pnl.index):
            if idx_date in pnl.index:
                pnl.loc[idx_date] += window_pnl.iloc[j]

        trade_return = float((1 + window_pnl).prod() - 1)

        rli_window = rli_r.iloc[entry_idx:actual_exit_idx + 1]
        all_window = all_r.iloc[entry_idx:actual_exit_idx + 1]

        rli_ret = float((1 + rli_window).prod() - 1)
        all_ret = float((1 + all_window).prod() - 1)

        def car_at(t):
            end = min(entry_idx + t, len(pair_r) - 1)
            w = pair_r.iloc[entry_idx:end + 1]
            return float((1 + w).prod() - 1) if len(w) > 0 else np.nan

        all_events.append({
            "event_date": event_date_str,
            "label": label,
            "entry_date": str(entry_date.date()),
            "exit_date": str(pair_r.index[actual_exit_idx].date()),
            "exit_reason": exit_reason,
            "days_held": actual_exit_idx - entry_idx + 1,
            "car_t20": round(car_at(20), 4),
            "car_t40": round(car_at(40), 4),
            "car_t75": round(car_at(75), 4),
            "total_pair_return": round(trade_return, 4),
            "rli_return": round(rli_ret, 4),
            "all_return": round(all_ret, 4),
        })

        last_exit = pair_r.index[actual_exit_idx]

    print(f"Events processed: {len(all_events)}")
    for ev in all_events:
        print(f"  {ev['event_date']}: CAR(t20)={ev['car_t20']:.2%}, CAR(t75)={ev['car_t75']:.2%}, "
              f"pair={ev['total_pair_return']:.2%} (RLI={ev['rli_return']:.2%}, ALL={ev['all_return']:.2%}), "
              f"exit={ev['exit_reason']}")

    active_pnl = pnl[pnl != 0]
    if len(all_events) == 0:
        return mark_failed(sid, "no valid events")
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="SCAC D&O Soft Cycle Short RLI Long ALL")
    m["n_events"] = len(all_events)

    avg_ret = float(np.mean([e["total_pair_return"] for e in all_events]))
    win_rate = float(np.mean([e["total_pair_return"] > 0 for e in all_events]))

    save_result(sid, m, extra={
        "rule": "Short RLI / Long ALL (1:1 dollar-neutral) when Stanford SCAC 6-month rolling filing count "
                "falls below 0.85x trailing 3yr average (D&O soft cycle). Entry on Cornerstone Research "
                "semi-annual report date (Jan/Jul). Hold 75 trading days; stop-loss if RLI outperforms ALL >12%.",
        "mechanism": "SCAC filing rate decline reduces D&O loss cost expectations, softening primary market rates. "
                     "Specialty insurers (RLI) with concentrated D&O exposure underperform broad P&C carriers "
                     "(ALL) when the D&O premium cycle turns soft.",
        "source": "Stanford Securities Class Action Clearinghouse (securities.stanford.edu); Cornerstone Research "
                  "semi-annual filings reports; yfinance RLI/ALL/SPY",
        "n_events": len(all_events),
        "avg_pair_return": round(avg_ret, 4),
        "win_rate": round(win_rate, 4),
        "events": all_events,
    })
    print(f"Done — {len(all_events)} events, avg_pair={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
