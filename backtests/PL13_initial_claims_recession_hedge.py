"""PL13_initial_claims_recession_hedge — Initial Claims Acceleration → Long TLT / Short XLY
When 4-week MA of initial claims rises >15% from 26-week low AND crosses 250k,
enter recession hedge pair. Exit when claims recover.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL13_initial_claims_recession_hedge"
    try:
        px = load_prices(["TLT", "XLY", "SPY"], start="2003-01-01")
        icsa = load_fred("ICSA", start="2002-06-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Compute 4-week MA of initial claims
    icsa_clean = icsa.dropna()
    icsa_4wk = icsa_clean.rolling(4).mean()
    icsa_26wk_low = icsa_4wk.rolling(26).min()

    # Signal: 4wk MA > 1.15 * 26wk low AND > 250000
    signal = (icsa_4wk > icsa_26wk_low * 1.15) & (icsa_4wk > 250000)
    # Exit: 4wk MA < 1.10 * 26wk low (recovery)
    exit_signal = icsa_4wk < icsa_26wk_low * 1.10

    # Build daily position series
    # Map weekly claims signal to daily positions
    in_position = False
    positions = pd.Series(0.0, index=ret.index)
    entry_dates = []
    exit_dates = []

    for date in signal.index:
        if not in_position and bool(signal.loc[date]):
            in_position = True
            entry_dates.append(date)
        elif in_position and date in exit_signal.index and bool(exit_signal.loc[date]):
            in_position = False
            exit_dates.append(date)

    # Fill daily positions from entry/exit dates
    for i, entry in enumerate(entry_dates):
        exit_d = exit_dates[i] if i < len(exit_dates) else ret.index[-1]
        mask = (ret.index >= entry) & (ret.index <= exit_d)
        positions[mask] = 1.0

    if positions.sum() == 0:
        return mark_failed(sid, "No signal triggers found")

    # Pair PnL: long TLT + short XLY when in position
    pair_daily = ret["TLT"] - ret["XLY"]
    pnl = pair_daily * positions

    # In-position only for metrics
    in_pos = pnl[positions > 0].dropna()
    if len(in_pos) < 30:
        return mark_failed(sid, f"Only {len(in_pos)} in-position days")

    # Build event list
    events = []
    for i, entry in enumerate(entry_dates):
        exit_d = exit_dates[i] if i < len(exit_dates) else ret.index[-1]
        mask = (ret.index >= entry) & (ret.index <= exit_d)
        window_pnl = pair_daily[mask]
        cum_ret = float((1 + window_pnl).prod() - 1)
        spy_cum = float((1 + spy_r[mask]).prod() - 1)
        days = int(mask.sum())
        events.append({
            "entry": str(entry.date()),
            "exit": str(exit_d.date()),
            "days": days,
            "pair_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(cum_ret - spy_cum, 4),
        })

    m = compute_metrics(in_pos, benchmark=spy_r[positions > 0].dropna(),
                        name="Claims Recession Hedge (TLT/XLY)")
    save_result(sid, m, extra={
        "rule": "ICSA 4wk MA > 1.15*26wk low AND > 250k → long TLT / short XLY; exit when 4wk MA < 1.10*26wk low",
        "mechanism": "Rising initial claims signal labor market deterioration → bonds rally on rate-cut expectations, cyclical consumer sells off",
        "source": "FRED ICSA + yfinance",
        "n_events": len(events),
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
