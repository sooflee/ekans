"""PL858_45v_finalrule_apd_cmi_long — 45V Treasury Final-Rule Policy Relaxation -> Long APD + CMI Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL858_45v_finalrule_apd_cmi_long"

    # Known event dates from the strategy spec:
    # Positive anchors: 2022-07-27 (IRA Senate reconciliation), 2023-10-13 (DOE H2Hub interim awards)
    # Negative control: 2023-12-22 (45V NPRM restrictive language)
    positive_events = [
        {"date": "2022-07-27", "label": "IRA Senate reconciliation vote (positive H2 policy)"},
        {"date": "2023-10-13", "label": "DOE H2Hub interim awards (positive H2 policy)"},
    ]
    negative_events = [
        {"date": "2023-12-22", "label": "45V NPRM restrictive three-pillar language"},
    ]

    try:
        px = load_prices(["APD", "CMI", "LIN", "PLUG", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or len(px) < 252:
        return mark_failed(sid, "insufficient price data")

    ret = daily_returns(px)
    apd_r = ret["APD"]
    cmi_r = ret["CMI"]
    spy_r = ret["SPY"]

    # Strategy: equal-dollar APD:CMI 60:40 weighting, short 20% SPY hedge
    # pair_return = 0.6 * APD + 0.4 * CMI - 0.2 * SPY
    pair_r = 0.6 * apd_r + 0.4 * cmi_r - 0.2 * spy_r

    hold_weeks_max = 16
    hold_days_max = hold_weeks_max * 5  # ~80 trading days
    stop_loss_pct = 0.08  # 8% combined loss vs SPY
    target_excess = 0.15  # +15% combined excess vs SPY
    early_exit_days = 20  # 4 weeks for stop-loss check

    all_events = []
    pnl = pd.Series(0.0, index=pair_r.index)
    last_exit = pd.Timestamp("2000-01-01")

    for ev in positive_events:
        trigger = pd.Timestamp(ev["date"])

        # Entry: next market open after trigger date
        next_dates = pair_r.index[pair_r.index > trigger]
        if len(next_dates) == 0:
            continue
        entry_date = next_dates[0]
        if entry_date <= last_exit:
            continue

        entry_idx = pair_r.index.get_loc(entry_date)

        # Exit conditions:
        # 1. max hold period (80 trading days)
        # 2. combined excess > 15% (but pair_r already nets out 20% SPY)
        # 3. stop-loss: pair underperforms by 8% in first 4 weeks
        actual_exit_idx = min(entry_idx + hold_days_max, len(pair_r) - 1)

        # Cumulative pair return to check stop/target
        cum_ret = 0.0
        exit_reason = "max_hold"
        for j in range(entry_idx, actual_exit_idx + 1):
            daily = pair_r.iloc[j]
            if np.isnan(daily):
                continue
            cum_ret = (1 + cum_ret) * (1 + daily) - 1
            days_in = j - entry_idx + 1

            # Check stop loss within first 4 weeks
            if days_in <= early_exit_days and cum_ret < -stop_loss_pct:
                actual_exit_idx = j
                exit_reason = "stop_loss"
                break
            # Check profit target
            if cum_ret > target_excess:
                actual_exit_idx = j
                exit_reason = "profit_target"
                break

        window_pnl = pair_r.iloc[entry_idx:actual_exit_idx + 1]
        pnl.iloc[entry_idx:actual_exit_idx + 1] = window_pnl.values

        # Raw APD+CMI excess return vs SPY over same window
        apd_window = apd_r.iloc[entry_idx:actual_exit_idx + 1]
        cmi_window = cmi_r.iloc[entry_idx:actual_exit_idx + 1]
        spy_window = spy_r.iloc[entry_idx:actual_exit_idx + 1]

        pair_return = float((1 + window_pnl).prod() - 1)
        spy_return_raw = float((1 + spy_window).prod() - 1)
        car = pair_return  # already excess

        # CAR at t+5, t+20, t+60
        def car_at(t):
            end = min(entry_idx + t, len(pair_r) - 1)
            w = pair_r.iloc[entry_idx:end + 1]
            return float((1 + w).prod() - 1)

        all_events.append({
            "event_date": ev["date"],
            "label": ev["label"],
            "entry_date": str(entry_date.date()),
            "exit_date": str(pair_r.index[actual_exit_idx].date()),
            "exit_reason": exit_reason,
            "days_held": actual_exit_idx - entry_idx + 1,
            "car_t5": round(car_at(5), 4),
            "car_t20": round(car_at(20), 4),
            "car_t60": round(car_at(60), 4),
            "total_return": round(pair_return, 4),
            "spy_return": round(spy_return_raw, 4),
            "event_type": "positive",
        })
        last_exit = pair_r.index[actual_exit_idx]

    # Negative control events — check CAR was negative
    for ev in negative_events:
        trigger = pd.Timestamp(ev["date"])
        next_dates = pair_r.index[pair_r.index > trigger]
        if len(next_dates) == 0:
            continue
        entry_date = next_dates[0]
        entry_idx = pair_r.index.get_loc(entry_date)

        def car_at_neg(t):
            end = min(entry_idx + t, len(pair_r) - 1)
            w = pair_r.iloc[entry_idx:end + 1]
            return float((1 + w).prod() - 1)

        all_events.append({
            "event_date": ev["date"],
            "label": ev["label"],
            "car_t5": round(car_at_neg(5), 4),
            "car_t20": round(car_at_neg(20), 4),
            "car_t60": round(car_at_neg(60), 4),
            "event_type": "negative_control",
        })

    print(f"Events processed: {len(all_events)}")
    for ev in all_events:
        if ev["event_type"] == "positive":
            print(f"  [+] {ev['event_date']}: CAR(t5)={ev['car_t5']:.2%}, CAR(t20)={ev['car_t20']:.2%}, "
                  f"CAR(t60)={ev['car_t60']:.2%}, total={ev['total_return']:.2%}, exit={ev['exit_reason']}")
        else:
            print(f"  [-] {ev['event_date']} (neg ctrl): CAR(t5)={ev['car_t5']:.2%}, CAR(t20)={ev['car_t20']:.2%}, "
                  f"CAR(t60)={ev['car_t60']:.2%}")

    # Check signal strength rule: CAR(t+20) > 0 in both positive anchors AND < 0 in negative control
    pos_events = [e for e in all_events if e["event_type"] == "positive"]
    neg_events = [e for e in all_events if e["event_type"] == "negative_control"]

    signal_valid = (
        all(e["car_t20"] > 0 for e in pos_events) and
        all(e["car_t20"] < 0 for e in neg_events)
    )
    print(f"Signal strength rule holds: {signal_valid}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)}) — only {len(pos_events)} positive events")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="45V Final Rule APD+CMI Long Pair")
    m["n_events"] = len(pos_events)

    save_result(sid, m, extra={
        "rule": "Long APD 60% + CMI 40% - short SPY 20% at next open after a 45V final-rule relaxation event "
                "(score >= 2 on temporal-matching / ERCOT zone / PPA grandfathering pillars). "
                "Hold up to 16 weeks; exit early on +15% excess gain or -8% stop-loss.",
        "mechanism": "Permissive 45V implementation lowers additionality/deliverability hurdles, directly "
                     "expanding the economics of APD's H2 projects and CMI Accelera's PEM electrolyzer order book. "
                     "Rate-relaxation reduces capex risk premium embedded in both stocks.",
        "source": "yfinance APD/CMI/LIN/PLUG/SPY; event dates from Federal Register / DOE announcements",
        "signal_valid": signal_valid,
        "n_events": len(pos_events),
        "events": all_events,
    })
    print(f"Done — {len(pos_events)} positive events")


if __name__ == "__main__":
    main()
