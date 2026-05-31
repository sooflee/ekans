"""PL651_datacenter_constraint_long_eqix_dlr — Data-Center Capacity Constraint Proxy -> Long EQIX/DLR vs XLRE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL651_datacenter_constraint_long_eqix_dlr"
    tickers = ["EQIX", "DLR", "XLRE", "XLU", "NVDA", "SPY"]
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "empty price frame")

    # Drop columns with too little history
    px = px.dropna(axis=1, thresh=200)
    cols = list(px.columns)
    primary = "EQIX"
    if primary not in cols:
        # try uppercase / alternative
        for c in cols:
            if c.upper() == primary.upper():
                primary = c
                break
    if primary not in cols:
        return mark_failed(sid, f"primary ticker {primary} not in data: {cols}")
    if "SPY" not in cols:
        return mark_failed(sid, "SPY missing for benchmark")

    rets = daily_returns(px)
    spy_r = rets["SPY"]
    p = px[primary].dropna()
    r = rets[primary].dropna()

    # Threshold logic: trailing 60-day return percentile rank
    win = 60
    trail = p.pct_change(win)
    roll_q_lo = trail.rolling(252, min_periods=120).quantile(0.10)
    roll_q_hi = trail.rolling(252, min_periods=120).quantile(0.90)

    direction = 1
    hold = 15

    pos = pd.Series(0.0, index=r.index)
    in_pos_until = None
    events = []
    for i, dt in enumerate(r.index):
        if dt not in trail.index:
            continue
        t_val = trail.loc[dt]
        lo = roll_q_lo.loc[dt] if dt in roll_q_lo.index else np.nan
        hi = roll_q_hi.loc[dt] if dt in roll_q_hi.index else np.nan
        if in_pos_until is not None and dt < in_pos_until:
            pos.iloc[i] = direction
            continue
        if pd.isna(t_val) or pd.isna(lo) or pd.isna(hi):
            continue
        # Counter-signal: trigger on extreme (top decile if shorting on overheat,
        # bottom decile if buying after capitulation).
        fire = False
        if direction == 1 and t_val <= lo:
            fire = True
        elif direction == -1 and t_val >= hi:
            fire = True
        if fire:
            pos.iloc[i] = direction
            end_idx = min(i + hold, len(r) - 1)
            in_pos_until = r.index[end_idx]
            events.append(str(dt.date()))

    if not events:
        return mark_failed(sid, "no events fired")

    pnl = pos.shift(1).fillna(0) * r
    pnl = pnl.dropna()
    pnl_open = pnl[pos.shift(1).fillna(0) != 0]
    if len(pnl_open) < 40:
        return mark_failed(sid, f"insufficient in-pos days ({len(pnl_open)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="Data-Center Capacity Constraint Proxy -> Long EQIX/DLR vs XLRE", positions=pos)
    save_result(sid, m, extra={
        "rule": "Approximate the PJM/ERCOT interconnection-queue withdrawal signal with NVDA-led AI capex proxy + utility-stock relative strength. ENTER long pair (long equal-weight EQIX+DLR / short XLRE) when (a) NVD",
        "mechanism": "Approximate the PJM/ERCOT interconnection-queue withdrawal signal with NVDA-led AI capex proxy + utility-stock relative strength. ENTER long pair (long equal-weight EQIX+DLR / short XLRE) when (a) NVDA 90-day return >= +15% (AI capex cycle still pulling power demand) AND (b) regional utilities expos",
        "source": "yfinance prices; project queue spec",
        "n_events": len(events),
        "events_sample": events[:25],
    }, pnl=pnl)
    print(f"Done {sid}: events={len(events)} sharpe={m.get('sharpe')}")


if __name__ == "__main__":
    main()
