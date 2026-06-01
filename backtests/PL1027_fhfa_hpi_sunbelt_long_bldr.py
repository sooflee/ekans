"""PL1027 — FHFA HPI Acceleration in 3+ Sun Belt States → Long BLDR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1027_fhfa_hpi_sunbelt_long_bldr"

    # FHFA state HPI series (quarterly)
    state_series = ["FLSTHPI", "NCSTHPI", "TXSTHPI", "AZSTHPI", "COSTHPI", "TNSTHPI"]

    try:
        hpi_data = {}
        for s in state_series:
            hpi_data[s] = load_fred(s, start="2015-01-01").squeeze()
        permits = load_fred("PERMIT1", start="2015-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    # Compute QoQ annualized rate and acceleration for each state
    # Quarterly data: resample to Q-end if needed, then compute QoQ and diff
    state_signals = {}
    for s in state_series:
        series = hpi_data[s].dropna()
        if len(series) < 4:
            print(f"Warning: {s} has too few data points")
            continue
        # Resample to quarterly if not already
        q_series = series.resample("QE").last().dropna()
        # QoQ annualized growth rate (quarterly rate * 4)
        qoq_ann = q_series.pct_change() * 4
        # Acceleration = change in QoQ annualized rate vs prior quarter
        accel = qoq_ann.diff()
        state_signals[s] = {"qoq_ann": qoq_ann, "accel": accel}

    if len(state_signals) < 3:
        return mark_failed(sid, "insufficient state HPI data")

    # PERMIT1: monthly, compute YoY
    permits_ff = permits.ffill().dropna()
    permits_yoy = permits_ff.pct_change(12)

    # Find all quarter-end dates where:
    # - 3+ states have QoQ annualized acceleration >= +5% (0.05 as fraction)
    # - PERMIT1 YoY >= -5%
    all_q_dates = sorted(set(
        d for s in state_signals.values()
        for d in s["accel"].index
    ))

    triggers = []
    last_trigger = None
    COOLDOWN_DAYS = 90  # no re-entry within 90 days

    for q_date in all_q_dates:
        if last_trigger is not None and (q_date - last_trigger).days < COOLDOWN_DAYS:
            continue

        # Count states with acceleration >= 5% annualized
        n_accel = 0
        for s, data in state_signals.items():
            if q_date in data["accel"].index:
                acc_val = float(data["accel"].loc[q_date])
                if not np.isnan(acc_val) and acc_val >= 0.05:
                    n_accel += 1

        if n_accel < 3:
            continue

        # Check PERMIT1 condition
        # Find nearest permit observation at or before q_date
        perm_before = permits_yoy[permits_yoy.index <= q_date]
        if perm_before.empty:
            continue
        perm_yoy = float(perm_before.iloc[-1])
        if np.isnan(perm_yoy) or perm_yoy < -0.05:
            continue

        # Signal fires with ~2 month data lag (FHFA releases ~2 months after Q end)
        # Add ~10 trading days extra for release + processing lag
        signal_date = q_date + pd.DateOffset(months=2, weeks=2)
        triggers.append({
            "q_date": q_date,
            "signal_date": signal_date,
            "n_states_accelerating": n_accel,
            "permit_yoy": round(perm_yoy, 4),
        })
        last_trigger = q_date

    print(f"Signal dates: {[(str(t['q_date'].date()), t['n_states_accelerating']) for t in triggers]}")

    if not triggers:
        return mark_failed(sid, "no FHFA HPI acceleration events found")

    try:
        px = load_prices(["BLDR", "BECN", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    bldr_r = ret["BLDR"]
    spy_r = ret["SPY"]
    hold_td = 63  # trading days (~3 months)

    pnl = pd.Series(0.0, index=bldr_r.index)
    events = []

    for trig in triggers:
        sig_date = trig["signal_date"]
        future_mask = bldr_r.index >= sig_date
        if future_mask.sum() < 10:
            continue
        entry_idx = bldr_r.index[future_mask][0]
        entry_pos = bldr_r.index.get_loc(entry_idx)
        exit_pos = min(entry_pos + hold_td, len(bldr_r))

        window = bldr_r.iloc[entry_pos:exit_pos]

        # Apply stops: -12% cumulative stop, +25% take profit
        cum = 1.0
        actual_exit = exit_pos
        stop_type = None
        for j, r in enumerate(window):
            cum *= (1 + r)
            if cum < 0.88:
                actual_exit = entry_pos + j + 1
                stop_type = "stop_loss"
                break
            elif cum > 1.25:
                actual_exit = entry_pos + j + 1
                stop_type = "take_profit"
                break

        window_actual = bldr_r.iloc[entry_pos:actual_exit]
        cum_ret = float((1 + window_actual).prod() - 1)

        # SPY over same window
        spy_future = spy_r.index >= sig_date
        spy_cum = None
        if spy_future.sum() >= 5:
            sp_entry = spy_r.index[spy_future][0]
            sp_pos = spy_r.index.get_loc(sp_entry)
            sp_win = spy_r.iloc[sp_pos: sp_pos + len(window_actual)]
            spy_cum = float((1 + sp_win).prod() - 1)

        # Write to pnl series
        pnl.iloc[entry_pos:actual_exit] = window_actual.values[: actual_exit - entry_pos]

        events.append({
            "q_date": str(trig["q_date"].date()),
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "n_states": trig["n_states_accelerating"],
            "permit_yoy": trig["permit_yoy"],
            "bldr_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "exit_type": stop_type or "hold_expiry",
        })

    if not events:
        return mark_failed(sid, "no valid events after data filtering")

    active = pnl[pnl != 0]
    print(f"Active trading days: {len(active)}")
    print(f"Events: {events}")

    if len(active) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active)}) — only {len(events)} event(s)")

    m = compute_metrics(active, benchmark=spy_r, name="FHFA HPI Sun Belt Acceleration → Long BLDR")
    bldr_rets = [e["bldr_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long BLDR 63 trading days when 3+ Sun Belt states (FL/NC/TX/AZ/CO/TN) show >=5% QoQ annualized HPI acceleration and PERMIT1 YoY >= -5%",
        "mechanism": "Sun Belt HPI acceleration signals regional wealth effect and new construction boom; BLDR (building materials distributor) captures ~70% revenue from new residential construction with heavy Sun Belt concentration",
        "source": "FRED FLSTHPI/NCSTHPI/TXSTHPI/AZSTHPI/COSTHPI/TNSTHPI (FHFA quarterly), PERMIT1; yfinance BLDR",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(bldr_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in bldr_rets])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}")


if __name__ == "__main__":
    main()
