"""PL1040 — Drewry WCI Asia-US Spot Rate Spike -> Short XRT Import Retailers (Counter-Signal)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1040_drewry_wci_spike_short_xrt"

    # Drewry WCI data is not on FRED/yfinance; we use known event dates as stated in strategy spec.
    # Known trigger events (from strategy dev notes): WCI Asia-US rose >40% in 8-week rolling window
    # Using the known_events from strategy + implementation notes for hardcoded event-study approach.
    # These are the dates WCI crossed the +40% 8-week threshold:
    KNOWN_TRIGGER_DATES = [
        pd.Timestamp("2021-06-03"),   # COVID port congestion surge
        pd.Timestamp("2021-10-07"),   # Second COVID surge
        pd.Timestamp("2023-11-30"),   # Red Sea crisis onset
        pd.Timestamp("2024-01-15"),   # Red Sea continuation
    ]

    try:
        px = load_prices(["XRT", "XLP", "SPY"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    xrt_r = ret["XRT"]
    xlp_r = ret["XLP"]
    spy_r = ret["SPY"]
    hold_td = 50  # trading days (~10 weeks)

    # PnL = short XRT (1x) + long XLP (0.5x)
    pnl = pd.Series(0.0, index=xrt_r.index)
    events = []

    # Apply cooldown to avoid overlapping positions from same episode
    last_exit_date = None
    COOLDOWN_DAYS = 30

    for sig_date in KNOWN_TRIGGER_DATES:
        # Check cooldown
        if last_exit_date is not None and (sig_date - last_exit_date).days < COOLDOWN_DAYS:
            print(f"Skipping {sig_date.date()} — still in cooldown after {last_exit_date.date()}")
            continue

        # Entry: 2 trading days after signal confirmation
        future_mask = xrt_r.index >= sig_date
        if future_mask.sum() < 5:
            continue
        # Skip 2 trading days from signal
        candidate_dates = xrt_r.index[future_mask]
        if len(candidate_dates) < 3:
            continue
        entry_idx = candidate_dates[min(2, len(candidate_dates) - 1)]
        entry_pos = xrt_r.index.get_loc(entry_idx)
        exit_pos = min(entry_pos + hold_td, len(xrt_r))

        # Check: XRT has NOT already corrected >10% from signal date to entry
        sig_future = xrt_r.index >= sig_date
        sig_pos = xrt_r.index.get_loc(xrt_r.index[sig_future][0])
        xrt_sig_to_entry = xrt_r.iloc[sig_pos:entry_pos]
        cum_xrt_to_entry = float((1 + xrt_sig_to_entry).prod() - 1) if len(xrt_sig_to_entry) > 0 else 0.0
        if cum_xrt_to_entry < -0.10:
            print(f"Skipping {sig_date.date()} — XRT already corrected {cum_xrt_to_entry:.1%} before entry")
            continue

        window_xrt = xrt_r.iloc[entry_pos:exit_pos]
        window_xlp = xlp_r.iloc[entry_pos:exit_pos]
        window_spy = spy_r.iloc[entry_pos:exit_pos]

        # Apply stop-loss: XRT outperforms SPY by >10% cumulative from entry
        cum_xrt = 1.0
        cum_spy = 1.0
        actual_exit = exit_pos
        stop_type = None
        for j in range(len(window_xrt)):
            r_xrt = float(window_xrt.iloc[j])
            r_spy = float(window_spy.iloc[j]) if j < len(window_spy) else 0.0
            cum_xrt *= (1 + r_xrt)
            cum_spy *= (1 + r_spy)
            if cum_xrt - cum_spy > 0.10:  # XRT up >10% vs SPY (short losing)
                actual_exit = entry_pos + j + 1
                stop_type = "stop_loss"
                break

        window_xrt_act = xrt_r.iloc[entry_pos:actual_exit]
        window_xlp_act = xlp_r.iloc[entry_pos:actual_exit]
        window_spy_act = spy_r.iloc[entry_pos:actual_exit]

        # PnL series: -1.0 * XRT + 0.5 * XLP (normalized to 1.0 notional)
        pnl_series = -1.0 * window_xrt_act + 0.5 * window_xlp_act

        xrt_cum = float((1 + window_xrt_act).prod() - 1)
        xlp_cum = float((1 + window_xlp_act).prod() - 1)
        spy_cum = float((1 + window_spy_act).prod() - 1)
        trade_pnl = -xrt_cum + 0.5 * xlp_cum

        pnl.iloc[entry_pos:actual_exit] = pnl_series.values[: actual_exit - entry_pos]
        last_exit_date = xrt_r.index[actual_exit - 1] if actual_exit > entry_pos else sig_date

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "xrt_return": round(xrt_cum, 4),
            "xlp_return": round(xlp_cum, 4),
            "spy_return": round(spy_cum, 4),
            "trade_pnl": round(trade_pnl, 4),
            "exit_type": stop_type or "time_stop",
        })

    print(f"Events: {events}")

    if not events:
        return mark_failed(sid, "no valid events after filtering")

    active = pnl[pnl != 0]
    print(f"Active trading days: {len(active)}")

    if len(active) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active)}) — {len(events)} event(s)")

    m = compute_metrics(active, benchmark=spy_r, name="Drewry WCI Spike → Short XRT")
    trade_pnls = [e["trade_pnl"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short XRT (1x) + Long XLP (0.5x) for 10 weeks when Drewry WCI Asia-US rate rises >40% in 8-week rolling window",
        "mechanism": "Container freight spikes increase COGS for import-heavy retailers (XRT) with ~2-quarter lag; short XRT captures margin compression before analyst downgrades; XLP hedge for systemic risk",
        "source": "Drewry WCI (hardcoded known events); yfinance XRT, XLP, SPY",
        "n_events": len(events),
        "avg_trade_pnl": round(float(np.mean(trade_pnls)), 4),
        "event_win_rate": round(float(np.mean([p > 0 for p in trade_pnls])), 4),
        "events": events,
        "caveat": "Drewry WCI not on FRED/yfinance — backtest uses hardcoded known signal dates; not live-tradeable without WCI data feed",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}")


if __name__ == "__main__":
    main()
