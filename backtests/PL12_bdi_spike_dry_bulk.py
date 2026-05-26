"""PL12_bdi_spike_dry_bulk — BDI Spike → Long Dry Bulk Shippers
When BDI spikes >50% from 3mo low, spot-exposed shippers re-rate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL12_bdi_spike_dry_bulk"
    try:
        px = load_prices(["SBLK", "GOGL", "BDRY", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)

    # Use BDRY as BDI proxy
    bdry_px = px["BDRY"] if "BDRY" in px.columns else None
    if bdry_px is None or bdry_px.dropna().empty:
        # Try SBLK as a proxy for dry bulk sentiment
        return mark_failed(sid, "BDRY not available on yfinance")

    # Compute 60-day rolling min, check when current price > 1.5x the min
    bdry_roll_min = bdry_px.rolling(60).min()
    spike_signal = bdry_px / bdry_roll_min

    basket = (ret["SBLK"] + ret["GOGL"]) / 2
    spy_r = ret["SPY"]

    events = []
    pnl_parts = []
    hold_days = 60
    last_entry = None

    for date in spike_signal.dropna().index:
        ratio = float(spike_signal.loc[date])
        if np.isnan(ratio) or ratio <= 1.5:
            continue
        # Avoid overlapping entries (must be >60 days since last)
        if last_entry and (date - last_entry).days < 70:
            continue

        entry_mask = ret.index >= date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        daily_pnl = basket.iloc[window]
        pnl_parts.append(daily_pnl)

        cum_ret = float((1 + daily_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.iloc[window]).prod() - 1)
        events.append({
            "trigger_date": str(date.date()),
            "bdry_spike_ratio": round(ratio, 2),
            "basket_60d_return": round(cum_ret, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess": round(cum_ret - spy_cum, 4),
        })
        last_entry = date

    if not events:
        return mark_failed(sid, "No BDI spike events (BDRY +50% from 60d low)")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="BDI Spike → Long Dry Bulk")
    save_result(sid, m, extra={
        "rule": "BDRY +50% from 60-day rolling low → long SBLK+GOGL 60 days",
        "mechanism": "BDI spike = spot charter rates surge; spot-exposed shippers earn windfall; sell-side updates lag 2-4 weeks",
        "source": "yfinance BDRY as BDI proxy",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
