"""PL1036 — Continuing Claims 4-Week MA Inflection (+5% from Trough) → Short HYG Counter-Signal"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1036_continuing_claims_inflection_hyg_short"

    try:
        ccsa = load_fred("CCSA", start="2005-01-01").squeeze()
        oas = load_fred("BAMLH0A0HYM2", start="2005-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if ccsa.empty:
        return mark_failed(sid, "CCSA empty")

    # CCSA is weekly (Thursday); compute 4-week MA and rolling 52-week min
    ccsa = ccsa.dropna()
    ccsa_4wma = ccsa.rolling(window=4, min_periods=4).mean()
    ccsa_52wmin = ccsa_4wma.rolling(window=52, min_periods=26).min()

    # OAS: forward-fill to daily, then align with weekly CCSA
    oas = oas.ffill().dropna()

    # Find signal dates: first time 4wMA crosses >= 5% above 52wk min
    # (with cooldown to avoid re-triggering during same episode)
    triggers = []
    last_trigger_date = None
    COOLDOWN_DAYS = 120  # prevent re-triggering during same cycle

    # Exclude COVID spike (2020-02 to 2020-12)
    COVID_START = pd.Timestamp("2020-02-01")
    COVID_END = pd.Timestamp("2020-12-31")

    in_signal = False  # tracking if already in an elevated regime

    for i in range(1, len(ccsa_4wma)):
        d = ccsa_4wma.index[i]
        ma_val = float(ccsa_4wma.iloc[i])
        min_val = float(ccsa_52wmin.iloc[i])
        prev_ma = float(ccsa_4wma.iloc[i - 1])
        prev_min = float(ccsa_52wmin.iloc[i - 1])

        if np.isnan(ma_val) or np.isnan(min_val) or min_val == 0:
            continue

        # Skip COVID period
        if COVID_START <= d <= COVID_END:
            in_signal = False
            continue

        ratio = ma_val / min_val
        prev_ratio = prev_ma / prev_min if (not np.isnan(prev_min) and prev_min != 0) else 1.0

        # Detect when ratio first crosses 1.05 from below (edge trigger)
        if prev_ratio < 1.05 and ratio >= 1.05:
            # Check cooldown
            if last_trigger_date is not None and (d - last_trigger_date).days < COOLDOWN_DAYS:
                in_signal = True
                continue

            # Check OAS < 4% at signal (spreads still compressed)
            oas_before = oas[oas.index <= d]
            if oas_before.empty:
                continue
            oas_val = float(oas_before.iloc[-1])
            if np.isnan(oas_val) or oas_val >= 4.0:
                in_signal = True
                continue

            triggers.append({
                "signal_date": d,
                "ccsa_4wma": round(ma_val, 0),
                "ccsa_52wmin": round(min_val, 0),
                "ccsa_ratio": round(ratio, 4),
                "oas_at_signal": round(oas_val, 2),
            })
            last_trigger_date = d
            in_signal = True
        elif ratio < 1.02:
            in_signal = False

    print(f"Signal dates found: {[(str(t['signal_date'].date()), t['oas_at_signal']) for t in triggers]}")

    if not triggers:
        return mark_failed(sid, "no continuing claims inflection signals found (all filtered by OAS or COVID exclusion)")

    try:
        px = load_prices(["HYG", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    hyg_r = ret["HYG"]
    spy_r = ret["SPY"]
    hold_days = 40  # ~56 calendar days ≈ ~40 trading days

    pnl = pd.Series(0.0, index=hyg_r.index)
    events = []

    for trig in triggers:
        sig_date = trig["signal_date"]
        # Entry on next trading day (claims released Thursday, enter Friday or Monday)
        future_mask = hyg_r.index > sig_date
        if future_mask.sum() < 5:
            continue
        entry_idx = hyg_r.index[future_mask][0]
        entry_pos = hyg_r.index.get_loc(entry_idx)
        exit_pos = min(entry_pos + hold_days, len(hyg_r))

        window = hyg_r.iloc[entry_pos:exit_pos]

        # Short HYG: PnL = -1 * HYG return
        # Stop-loss: HYG closes > 3% above entry (i.e., we lose > 3% on the short)
        cum_hyg = 1.0
        actual_exit = exit_pos
        stop_type = None
        for j, r in enumerate(window):
            cum_hyg *= (1 + r)
            if cum_hyg > 1.03:  # HYG up >3%, short stops out
                actual_exit = entry_pos + j + 1
                stop_type = "stop_loss"
                break

        window_actual = hyg_r.iloc[entry_pos:actual_exit]
        hyg_cum = float((1 + window_actual).prod() - 1)
        short_pnl_series = -window_actual  # short position

        spy_future = spy_r.index >= sig_date
        spy_cum = None
        if spy_future.sum() >= 5:
            sp_entry = spy_r.index[spy_future][0]
            sp_pos = spy_r.index.get_loc(sp_entry)
            sp_win = spy_r.iloc[sp_pos: sp_pos + len(window_actual)]
            spy_cum = float((1 + sp_win).prod() - 1)

        # Write PnL (short returns)
        pnl.iloc[entry_pos:actual_exit] = short_pnl_series.values[: actual_exit - entry_pos]

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "oas_at_entry": trig["oas_at_signal"],
            "hyg_return": round(hyg_cum, 4),
            "short_pnl": round(-hyg_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "exit_type": stop_type or "time_stop",
        })

    if not events:
        return mark_failed(sid, "no valid events after data filtering")

    active = pnl[pnl != 0]
    print(f"Active trading days: {len(active)}")
    print(f"Events: {events}")

    if len(active) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active)}) — {len(events)} event(s)")

    m = compute_metrics(active, benchmark=spy_r, name="CCSA Inflection → Short HYG")
    short_pnls = [e["short_pnl"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short HYG for ~56 calendar days when CCSA 4wMA rises >=5% above 52wk minimum and HY OAS < 4%",
        "mechanism": "Labor market softening (rising continuing claims) precedes HY credit spread widening by ~6 weeks; short HYG captures spread widening before equity markets price it",
        "source": "FRED CCSA, BAMLH0A0HYM2; yfinance HYG",
        "n_events": len(events),
        "avg_short_pnl": round(float(np.mean(short_pnls)), 4),
        "event_win_rate": round(float(np.mean([p > 0 for p in short_pnls])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}")


if __name__ == "__main__":
    main()
