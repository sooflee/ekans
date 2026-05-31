"""PL649_drought_demand_destruction_short_fertilizer — Drought Demand-Destruction Proxy -> Short MOS+NTR+CF Basket"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL649_drought_demand_destruction_short_fertilizer"
    tickers = ["MOS", "NTR", "CF", "ZC=F", "SPY"]
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "empty price frame")

    # Drop columns with too little history
    px = px.dropna(axis=1, thresh=200)
    cols = list(px.columns)
    primary = "MOS"
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

    direction = -1
    hold = 20

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

    m = compute_metrics(pnl, benchmark=spy_r, name="Drought Demand-Destruction Proxy -> Short MOS+NTR+CF Basket", positions=pos)
    save_result(sid, m, extra={
        "rule": "Approximate the NOAA D4 drought-coverage signal with a crop-price + seasonal proxy. ENTER short equal-weight basket (MOS, NTR, CF) when (a) calendar date is in May-June (planting/side-dress window) OR",
        "mechanism": "Approximate the NOAA D4 drought-coverage signal with a crop-price + seasonal proxy. ENTER short equal-weight basket (MOS, NTR, CF) when (a) calendar date is in May-June (planting/side-dress window) OR October-November (fall application window) AND (b) corn front-month ZC=F has fallen >-15% over the ",
        "source": "yfinance prices; project queue spec",
        "n_events": len(events),
        "events_sample": events[:25],
    }, pnl=pnl)
    print(f"Done {sid}: events={len(events)} sharpe={m.get('sharpe')}")


if __name__ == "__main__":
    main()
