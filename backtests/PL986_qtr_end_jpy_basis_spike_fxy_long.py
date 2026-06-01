"""PL986_qtr_end_jpy_basis_spike_fxy_long — Quarter-End JPY Cross-Currency Basis Spike -> Long FXY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL986_qtr_end_jpy_basis_spike_fxy_long"

    try:
        px = load_prices(["FXY", "SPY"], start="2007-01-01")
        fred = load_fred("DEXJPUS", start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "FXY" not in px.columns:
        return mark_failed(sid, "FXY not available")

    ret = daily_returns(px)
    fxy_r = ret["FXY"]
    spy_r = ret["SPY"]

    usdjpy = fred.squeeze().dropna()

    # Generate all quarter-end dates from 2007 to end of available data
    # Quarter-ends: March 31, June 30, Sept 30, Dec 31
    start_year = 2007
    end_year = fxy_r.index[-1].year + 1
    quarter_ends = []
    for yr in range(start_year, end_year):
        for month in [3, 6, 9, 12]:
            quarter_ends.append(pd.Timestamp(yr, month, 1) + pd.offsets.MonthEnd(0))

    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for qe in quarter_ends:
        # Entry: 7 calendar days before quarter-end
        entry_cal = qe - pd.Timedelta(days=7)

        # Find last trading day on or before entry_cal
        entry_mask = fxy_r.index <= entry_cal
        if entry_mask.sum() == 0:
            continue
        entry_idx = fxy_r.index[entry_mask][-1]

        # Exit: 2nd trading day of new quarter
        exit_start = qe + pd.Timedelta(days=1)
        exit_mask = fxy_r.index >= exit_start
        if exit_mask.sum() < 2:
            continue
        exit_idx = fxy_r.index[exit_mask][1]  # 2nd trading day of new quarter

        # BOJ filter: DEXJPUS within 5% of 60-day MA at entry
        usdjpy_before = usdjpy[usdjpy.index <= entry_idx]
        if len(usdjpy_before) < 60:
            continue
        ma60 = float(usdjpy_before.iloc[-60:].mean())
        current = float(usdjpy_before.iloc[-1])
        if abs(current - ma60) / ma60 > 0.05:
            # Outside 5% band — skip this event (extreme BOJ intervention)
            continue

        # Compute FXY return from entry to exit
        fxy_window = fxy_r[(fxy_r.index >= entry_idx) & (fxy_r.index <= exit_idx)]
        if len(fxy_window) < 1:
            continue

        spy_window = spy_r[(spy_r.index >= entry_idx) & (spy_r.index <= exit_idx)]

        fxy_total = float((1 + fxy_window).prod() - 1)
        spy_total = float((1 + spy_window).prod() - 1)

        # Accumulate PnL
        for dt, val in fxy_window.items():
            if dt in pnl.index:
                pnl.loc[dt] += val

        events.append({
            "quarter_end": str(qe.date()),
            "entry_date": str(entry_idx.date()),
            "exit_date": str(exit_idx.date()),
            "fxy_return": round(fxy_total, 4),
            "spy_return": round(spy_total, 4),
            "n_days": len(fxy_window),
        })

    print(f"Events (after BOJ filter): {len(events)}")
    returns = [e["fxy_return"] for e in events]
    if events:
        print(f"  Mean FXY return: {np.mean(returns):.2%}, Win rate: {np.mean([r>0 for r in returns]):.0%}")

    if not events:
        return mark_failed(sid, "no valid events after BOJ filter")

    # Active PnL (non-zero days)
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Quarter-End JPY Basis: Long FXY")

    mean_fxy = float(np.mean(returns))
    win_rate = float(np.mean([r > 0 for r in returns]))

    save_result(sid, m, extra={
        "rule": "Long FXY from T-7 calendar days before quarter-end through T+2 trading days after; filter: DEXJPUS within 5% of 60-day MA",
        "mechanism": "Quarter-end USD funding demand spikes as banks window-dress balance sheets; Japanese firms repatriate earnings; cross-currency basis widens -> JPY strengthens vs USD",
        "source": "FXY yfinance; DEXJPUS FRED; systematic quarter-end calendar trigger",
        "n_events": len(events),
        "mean_fxy_return": round(mean_fxy, 4),
        "win_rate": round(win_rate, 4),
        "events": events[:20],  # First 20 for brevity
    })
    print(f"Done. Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}")


if __name__ == "__main__":
    main()
