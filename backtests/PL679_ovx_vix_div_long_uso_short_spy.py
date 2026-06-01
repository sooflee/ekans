"""PL679 — OVX-VIX Vol Divergence - Long USO Short SPY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL679_ovx_vix_div_long_uso_short_spy"

    try:
        # Load volatility indices (^OVX, ^VIX) and price tickers
        vol_px = load_prices(["^OVX", "^VIX"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"vol data load: {e}")

    try:
        price_px = load_prices(["USO", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if vol_px.empty or price_px.empty:
        return mark_failed(sid, "no data")

    # Align indices
    idx = vol_px.index.intersection(price_px.index)
    if len(idx) < 100:
        return mark_failed(sid, f"insufficient overlapping data: {len(idx)} days")

    vol_px = vol_px.reindex(idx)
    price_px = price_px.reindex(idx)

    # OVX and VIX levels
    ovx = vol_px["^OVX"].dropna() if "^OVX" in vol_px.columns else None
    vix = vol_px["^VIX"].dropna() if "^VIX" in vol_px.columns else None

    if ovx is None or vix is None or ovx.empty or vix.empty:
        return mark_failed(sid, "OVX or VIX data not available")

    # Align OVX and VIX
    common_idx = ovx.index.intersection(vix.index)
    ovx = ovx.reindex(common_idx)
    vix = vix.reindex(common_idx)

    gap = ovx - vix

    # Compute 5-year rolling 90th percentile for top-decile threshold
    gap_90pct = gap.rolling(window=252 * 5, min_periods=252).quantile(0.90)

    # USO 90-day SMA
    uso = price_px["USO"].reindex(common_idx)
    uso_sma90 = uso.rolling(90).mean()

    # Entry signal: OVX-VIX > top decile of 5y AND USO within 5% of 90d SMA
    # Use fixed threshold of 25 (top-decile heuristic from strategy spec)
    gap_threshold = 25.0
    uso_within_range = (uso >= uso_sma90 * 0.95) & (uso <= uso_sma90 * 1.05)
    signal = (gap > gap_threshold) & uso_within_range & uso_sma90.notna()

    # Build event list with cooldown (avoid overlapping)
    ret = daily_returns(price_px.reindex(common_idx))
    spy_r = ret["SPY"] if "SPY" in ret.columns else pd.Series(0.0, index=ret.index)
    uso_r = ret["USO"] if "USO" in ret.columns else pd.Series(0.0, index=ret.index)

    hold = 15  # trading days
    pnl = pd.Series(0.0, index=ret.index)
    event_records = []
    last_event_end = pd.Timestamp("2000-01-01")

    signal_dates = signal[signal].index

    for event_date in signal_dates:
        # Skip if within cooldown period (last event hasn't ended yet)
        if event_date <= last_event_end:
            continue

        mask = ret.index >= event_date
        if mask.sum() == 0:
            continue
        start_idx = ret.index[mask][0]
        p = ret.index.get_loc(start_idx)
        end_idx = min(p + hold, len(ret))

        window_ret = ret.iloc[p:end_idx]

        uso_win = uso_r.iloc[p:end_idx].fillna(0)
        spy_win = spy_r.iloc[p:end_idx].fillna(0)

        # Long USO, short SPY 1:1
        event_pnl = uso_win - spy_win

        pnl.iloc[p:end_idx] = event_pnl.values
        last_event_end = ret.index[end_idx - 1] if end_idx > 0 else event_date

        total_return = float((1 + event_pnl).prod() - 1)
        spy_total = float((1 + spy_win).prod() - 1)
        gap_val = float(gap.loc[event_date]) if event_date in gap.index else None

        event_records.append({
            "event_date": str(event_date.date()),
            "ovx_vix_gap": round(gap_val, 2) if gap_val is not None else None,
            "pnl_return": round(total_return, 4),
            "spy_return": round(spy_total, 4),
        })

    print(f"Total signal days: {signal.sum()}")
    print(f"Non-overlapping events: {len(event_records)}")
    if event_records:
        print(f"  First: {event_records[0]['event_date']}, Last: {event_records[-1]['event_date']}")
    print(f"  Win rate: {np.mean([e['pnl_return'] > 0 for e in event_records]):.2%}" if event_records else "")

    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="OVX-VIX Divergence Long USO Short SPY")
    save_result(sid, m, extra={
        "rule": "Long USO, short SPY (1:1) for 15 days when OVX-VIX gap > 25 points (top decile) and USO within 5% of 90d SMA",
        "mechanism": "Extreme oil-specific volatility premium vs equity vol signals oil-specific geopolitical/supply stress; crude likely to mean-revert or spike while equities face energy cost headwind",
        "source": "yfinance: ^OVX, ^VIX, USO, SPY",
        "n_events": len(event_records),
        "total_signal_days": int(signal.sum()),
        "events_sample": event_records[:10],
        "status": "ok",
    })
    print(f"Done: {len(event_records)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
