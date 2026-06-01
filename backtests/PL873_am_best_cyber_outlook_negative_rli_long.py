"""PL873_am_best_cyber_outlook_negative_rli_long — AM Best Cyber Outlook Negative -> Long RLI+CB vs Short ALL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL873_am_best_cyber_outlook_negative_rli_long"

    # Known event dates from strategy spec:
    # Primary anchor dates: approximate AM Best Cyber Market Segment Report publication dates
    # When AM Best publishes Negative/hardening cyber outlook -> long RLI+CB vs short ALL
    known_events = [
        ("2022-01-03", "Lloyd's cyber sublimit mandate announcement (effective 1/1/2023), "
                       "hard cyber market signal"),
        ("2022-09-01", "AM Best 2022 Cyber Market Report (hardening confirmation, Negative outlook)"),
        ("2024-09-03", "AM Best 2024 Cyber Market Report (Negative outlook reaffirmed)"),
    ]

    try:
        px = load_prices(["RLI", "CB", "ALL", "AIG", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or len(px) < 252:
        return mark_failed(sid, "insufficient price data")

    # Check data coverage
    for ticker in ["RLI", "CB", "ALL"]:
        if ticker not in px.columns or px[ticker].notna().mean() < 0.8:
            return mark_failed(sid, f"insufficient data for {ticker}")

    ret = daily_returns(px)
    rli_r = ret["RLI"]
    cb_r = ret["CB"]
    all_r = ret["ALL"]
    spy_r = ret["SPY"]

    # Strategy: Long RLI 60% + CB 40%, Short ALL 50% notional
    # Net pair = 0.6*RLI + 0.4*CB - 0.5*ALL
    pair_r = 0.6 * rli_r + 0.4 * cb_r - 0.5 * all_r

    hold_days_max = 80  # ~16 weeks in trading days
    target_excess = 0.15  # +15% excess vs ALL
    stop_loss = 0.10  # -10% within 4 weeks
    early_exit_days = 20  # 4 weeks for stop-loss check

    pnl = pd.Series(0.0, index=spy_r.index)
    all_events = []
    last_exit = pd.Timestamp("2000-01-01")

    for event_date_str, label in known_events:
        trigger = pd.Timestamp(event_date_str)

        # Entry: next trading day after trigger (or same day if published pre-market)
        avail_dates = pair_r.index[pair_r.index >= trigger]
        if len(avail_dates) == 0:
            print(f"  Skipping {event_date_str} — no trading days")
            continue
        entry_date = avail_dates[0]
        if entry_date <= last_exit:
            print(f"  Skipping {event_date_str} — overlaps previous position")
            continue

        entry_idx = pair_r.index.get_loc(entry_date)
        max_exit_idx = min(entry_idx + hold_days_max, len(pair_r) - 1)

        # Scan for early exit
        cum_pair = 0.0
        cum_rli_cb = 0.0  # track basket vs ALL for excess return
        actual_exit_idx = max_exit_idx
        exit_reason = "time_stop"

        for j in range(entry_idx, max_exit_idx + 1):
            pr = pair_r.iloc[j]
            if np.isnan(pr):
                continue
            days_in = j - entry_idx + 1
            cum_pair = (1 + cum_pair) * (1 + pr) - 1

            # Stop-loss: basket underperforms by 10% within first 4 weeks
            if days_in <= early_exit_days and cum_pair < -stop_loss:
                actual_exit_idx = j
                exit_reason = "stop_loss"
                break
            # Profit target: basket outperforms by 15%
            if cum_pair > target_excess:
                actual_exit_idx = j
                exit_reason = "profit_target"
                break

        window_pnl = pair_r.iloc[entry_idx:actual_exit_idx + 1]
        # Add to global pnl
        for j, idx_date in enumerate(window_pnl.index):
            if idx_date in pnl.index:
                pnl.loc[idx_date] += window_pnl.iloc[j]

        trade_return = float((1 + window_pnl).prod() - 1)

        # Individual ticker returns over window
        rli_window = rli_r.iloc[entry_idx:actual_exit_idx + 1]
        cb_window = cb_r.iloc[entry_idx:actual_exit_idx + 1]
        all_window = all_r.iloc[entry_idx:actual_exit_idx + 1]
        spy_window = spy_r.iloc[entry_idx:actual_exit_idx + 1]

        rli_ret = float((1 + rli_window).prod() - 1)
        cb_ret = float((1 + cb_window).prod() - 1)
        all_ret = float((1 + all_window).prod() - 1)
        spy_ret = float((1 + spy_window).prod() - 1)

        def car_at(t):
            end = min(entry_idx + t, len(pair_r) - 1)
            w = pair_r.iloc[entry_idx:end + 1]
            return float((1 + w).prod() - 1) if len(w) > 0 else np.nan

        all_events.append({
            "event_date": event_date_str,
            "label": label[:60],
            "entry_date": str(entry_date.date()),
            "exit_date": str(pair_r.index[actual_exit_idx].date()),
            "exit_reason": exit_reason,
            "days_held": actual_exit_idx - entry_idx + 1,
            "car_t20": round(car_at(20), 4),
            "car_t40": round(car_at(40), 4),
            "car_t80": round(car_at(80), 4),
            "total_pair_return": round(trade_return, 4),
            "rli_return": round(rli_ret, 4),
            "cb_return": round(cb_ret, 4),
            "all_return": round(all_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

        last_exit = pair_r.index[actual_exit_idx]

    print(f"Events processed: {len(all_events)}")
    for ev in all_events:
        print(f"  {ev['event_date']}: CAR(t20)={ev['car_t20']:.2%}, CAR(t40)={ev['car_t40']:.2%}, "
              f"CAR(t80)={ev['car_t80']:.2%}, RLI={ev['rli_return']:.2%}, CB={ev['cb_return']:.2%}, "
              f"ALL={ev['all_return']:.2%}, exit={ev['exit_reason']}")

    active_pnl = pnl[pnl != 0]
    if len(all_events) == 0:
        return mark_failed(sid, "no valid events")
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="AM Best Cyber Outlook Long RLI+CB vs ALL")
    m["n_events"] = len(all_events)

    avg_ret = float(np.mean([e["total_pair_return"] for e in all_events]))
    win_rate = float(np.mean([e["total_pair_return"] > 0 for e in all_events]))

    save_result(sid, m, extra={
        "rule": "Long RLI 60% + CB 40%, Short ALL 50% (net notional) when AM Best publishes Cyber Insurance "
                "Market Segment Report with Negative/hardening outlook. Entry at open on announcement day. "
                "Hold up to 16 weeks; profit target +15% vs ALL, stop-loss -10% within 4 weeks.",
        "mechanism": "AM Best Negative cyber outlook signals elevated ransomware frequency/severity, benefiting "
                     "specialty cyber insurers (RLI, CB) via premium hardening and improved combined ratios, "
                     "while broad P&C carriers (ALL) face flat cyber exposure. The pair isolates specialty re-rating.",
        "source": "AM Best Cyber Market Segment Reports (news.ambest.com); yfinance RLI/CB/ALL/SPY; "
                  "Lloyd's cyber sublimit announcement Jan 2022",
        "n_events": len(all_events),
        "avg_pair_return": round(avg_ret, 4),
        "win_rate": round(win_rate, 4),
        "events": all_events,
    })
    print(f"Done — {len(all_events)} events, avg_pair={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
