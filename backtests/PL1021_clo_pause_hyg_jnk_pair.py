"""PL1021_clo_pause_hyg_jnk_pair — NFCI Surge → CLO Demand Freeze → Long HYG / Short JNK"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1021_clo_pause_hyg_jnk_pair"

    # Load FRED NFCI (weekly)
    try:
        nfci_raw = load_fred("NFCI", start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED NFCI load: {e}")

    nfci = nfci_raw.squeeze().dropna()
    if nfci.empty:
        return mark_failed(sid, "NFCI data empty")

    # Load prices
    try:
        px = load_prices(["HYG", "JNK", "SPY"], start="2007-12-01")
    except Exception as e:
        try:
            px = load_prices(["HYG", "JNK", "SPY"], start="2007-12-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)
    hyg_r = ret["HYG"]
    jnk_r = ret["JNK"]
    spy_r = ret["SPY"]

    # Compute 4-week (20-trading-day) NFCI change using weekly data
    # NFCI is weekly; 4 weeks = 4 weekly observations
    nfci_4w_change = nfci.diff(4)  # change over 4 weekly observations

    # Find entry signals: 4-week NFCI change > 0.20
    # Use weekly dates; forward-fill to daily
    nfci_daily = nfci.reindex(hyg_r.index, method="ffill")
    nfci_4w_daily = nfci_daily.diff(20)  # 20 trading days ≈ 4 weeks

    signal_mask = nfci_4w_daily > 0.20
    signal_dates = hyg_r.index[signal_mask]

    # Find entry dates: first day of each signal cluster (min 20-day gap between entries)
    entry_dates = []
    last_exit = None
    i = 0
    while i < len(signal_dates):
        dt = signal_dates[i]
        if last_exit is None or (dt - last_exit).days >= 20:
            entry_dates.append(dt)
            # Find exit: NFCI 4w change drops below 0.05 OR 30 calendar days
            exit_dt = dt + pd.Timedelta(days=30)
            # Search for NFCI normalization
            future_mask = (hyg_r.index > dt) & (hyg_r.index <= exit_dt)
            future_dates = hyg_r.index[future_mask]
            actual_exit = exit_dt
            for fut_dt in future_dates:
                nfci_val = nfci_4w_daily.get(fut_dt, np.nan)
                if not np.isnan(nfci_val) and nfci_val < 0.05:
                    actual_exit = fut_dt
                    break
            last_exit = actual_exit
        i += 1

    print(f"NFCI signal dates: {len(signal_dates)}")
    print(f"Filtered entry dates: {len(entry_dates)}")

    if not entry_dates:
        return mark_failed(sid, "no qualifying NFCI entry events")

    # Build PnL: long HYG, short JNK (equal notional)
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []
    occupied = set()

    for entry_idx in entry_dates:
        # Exit: 30 calendar days or NFCI normalization
        exit_dt_hard = entry_idx + pd.Timedelta(days=30)
        future_mask = (hyg_r.index > entry_idx) & (hyg_r.index <= exit_dt_hard)
        future_dates = hyg_r.index[future_mask]

        actual_exit_idx = exit_dt_hard
        exit_reason = "hard_stop_30d"
        for fut_dt in future_dates:
            nfci_val = nfci_4w_daily.get(fut_dt, np.nan)
            if not np.isnan(nfci_val) and nfci_val < 0.05:
                actual_exit_idx = fut_dt
                exit_reason = "nfci_normalized"
                break

        window_mask = (hyg_r.index > entry_idx) & (hyg_r.index <= actual_exit_idx)
        if not window_mask.any():
            continue

        window_hyg = hyg_r[window_mask]
        window_jnk = jnk_r.reindex(window_hyg.index).fillna(0.0)
        window_spy = spy_r.reindex(window_hyg.index).fillna(0.0)

        # Pair PnL: long HYG, short JNK
        daily_pair = window_hyg - window_jnk

        for idx, val in daily_pair.items():
            if idx not in occupied:
                pnl[idx] += val
                occupied.add(idx)

        total_hyg = float((1 + window_hyg).prod() - 1)
        total_jnk = float((1 + window_jnk).prod() - 1)
        total_spy = float((1 + window_spy).prod() - 1)
        pair_ret = total_hyg - total_jnk
        n_days = len(window_hyg)

        nfci_entry_val = float(nfci_4w_daily.get(entry_idx, np.nan))
        print(f"  Entry={entry_idx.date()} NFCI4w={nfci_entry_val:.3f}, "
              f"exit={actual_exit_idx.date()} ({exit_reason}): "
              f"HYG={total_hyg:.4f}, JNK={total_jnk:.4f}, pair={pair_ret:.4f}")

        events.append({
            "entry_date": str(entry_idx.date()),
            "exit_date": str(actual_exit_idx.date()),
            "exit_reason": exit_reason,
            "n_days_held": n_days,
            "nfci_4w_change_at_entry": round(nfci_entry_val, 4),
            "hyg_return": round(total_hyg, 4),
            "jnk_return": round(total_jnk, 4),
            "pair_return": round(pair_ret, 4),
            "spy_return": round(total_spy, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events after filtering")

    active_pnl = pnl[pnl != 0.0]
    print(f"\nEvents: {len(events)}, Active days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NFCI Surge → Long HYG / Short JNK")
    avg_pair = float(np.mean([e["pair_return"] for e in events]))
    win_rate = float(np.mean([e["pair_return"] > 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long HYG / Short JNK equal-notional when FRED NFCI 4-week change > 0.20; exit on NFCI normalization (<0.05) or 30-day hard stop",
        "mechanism": "Rapid NFCI tightening proxies CLO new-issue pause; JNK (higher cyclical/leveraged exposure) underperforms HYG during CLO demand freeze due to composition differences",
        "source": "FRED NFCI; yfinance HYG, JNK, SPY",
        "n_events": len(events),
        "avg_pair_return": round(avg_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: {len(events)} events")


if __name__ == "__main__":
    main()
