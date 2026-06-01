"""PL780_brent_ovx_spike_xlp_xly_rotation
Brent Backwardation + OVX Spike -> Long XLP / Short XLY Consumer Rotation

Trigger: OVX > 50 AND Brent (BZ=F) 20-day return > +15% on same day.
Enter long XLP, short XLY equal-dollar. Exit when OVX < 35 or 30-trading-day max.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL780_brent_ovx_spike_xlp_xly_rotation"

    try:
        px_eq = load_prices(["XLP", "XLY", "SPY"], start="2010-01-01")
    except Exception as e:
        try:
            px_eq = load_prices(["XLP", "XLY", "SPY"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"equity data load: {e2}")

    try:
        px_brent = load_prices(["BZ=F"], start="2010-01-01")
    except Exception as e:
        try:
            px_brent = load_prices(["BZ=F"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"brent data load: {e2}")

    try:
        px_ovx = load_prices(["^OVX"], start="2010-01-01")
    except Exception as e:
        try:
            px_ovx = load_prices(["^OVX"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"OVX data load: {e2}")

    # Validate required tickers
    for t in ["XLP", "XLY", "SPY"]:
        if t not in px_eq.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    if "BZ=F" not in px_brent.columns:
        return mark_failed(sid, "missing BZ=F ticker")

    if "^OVX" not in px_ovx.columns:
        return mark_failed(sid, "missing ^OVX ticker")

    # Combine on common index
    px = px_eq.join(px_brent, how="outer").join(px_ovx, how="outer")
    px = px.sort_index().ffill(limit=3)

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Signals
    brent = px["BZ=F"].dropna()
    ovx = px["^OVX"].dropna()

    # 20-day Brent return
    brent_ret_20 = brent.pct_change(20)

    # Align signals on common date range
    common_idx = brent_ret_20.dropna().index.intersection(ovx.dropna().index)
    b20 = brent_ret_20.reindex(common_idx)
    ov = ovx.reindex(common_idx)

    # Entry signal: OVX > 50 AND Brent 20d return > 15%
    entry_signal = (ov > 50) & (b20 > 0.15)

    # Build positions
    xlp_ret = ret["XLP"]
    xly_ret = ret["XLY"]

    max_hold = 30  # trading days
    positions = pd.Series(0.0, index=ret.index)

    in_pos = False
    entry_day_idx = None
    hold_count = 0

    idx_list = list(ret.index)

    for i, d in enumerate(idx_list):
        if in_pos:
            hold_count += 1
            # Exit conditions: OVX < 35, or max hold
            current_ovx = ov.get(d, np.nan) if d in ov.index else np.nan
            if (not np.isnan(current_ovx) and current_ovx < 35) or hold_count >= max_hold:
                in_pos = False
                hold_count = 0
                entry_day_idx = None
            else:
                positions.iloc[i] = 1.0  # long XLP-short XLY

        if not in_pos and d in entry_signal.index and entry_signal.get(d, False):
            in_pos = True
            hold_count = 0
            entry_day_idx = i

    # Pair PnL: long XLP - short XLY, positions applied to next day's return (shift)
    pair_ret = xlp_ret - xly_ret
    pair_ret = pair_ret.reindex(ret.index).fillna(0)

    pnl = positions.shift(1).fillna(0) * pair_ret

    # Align with SPY
    spy_aligned = spy_r.dropna()
    pnl_aligned = pnl.reindex(spy_aligned.index).fillna(0)

    n_events = int(entry_signal.sum())
    active_days = int((positions != 0).sum())

    m = compute_metrics(pnl_aligned, benchmark=spy_aligned, name="Brent+OVX Spike XLP/XLY Rotation")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "OVX > 50 AND Brent 20-day return > 15% -> long XLP, short XLY. Exit when OVX < 35 or 30-day max hold.",
        "mechanism": "Oil price shock raises gasoline pass-through costs, compressing consumer discretionary spending; Kilian (2009) 3-6 week transmission. Staples outperform discretionary in energy-shock regimes.",
        "source": "OVX ^OVX (yfinance), Brent BZ=F (yfinance), XLP/XLY (yfinance). Kilian (2009) J Economic Literature.",
        "status": "ok",
        "n_trigger_events": n_events,
        "active_days": active_days,
    }, pnl=pnl_aligned)

    print(f"\n=== {sid} ===")
    print(f"Trigger events: {n_events}, Active days: {active_days}")
    print(f"Sharpe: {m.get('sharpe', 'N/A'):.3f}  CAGR: {m.get('cagr', 0):.2%}  MaxDD: {m.get('max_dd', 0):.2%}  t-stat: {m.get('t_stat', 0):.3f}")
    if 'oos_sharpe' in m:
        print(f"OOS Sharpe: {m['oos_sharpe']:.3f}")


if __name__ == "__main__":
    main()
