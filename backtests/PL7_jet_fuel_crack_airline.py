"""PL7_jet_fuel_crack_airline — Jet Fuel Crack Spread → Short Airlines vs Long XLE
When jet fuel crack (proxy: WTI 3mo change vs airline stock 3mo change divergence)
spikes, short airlines / long energy for 60 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL7_jet_fuel_crack_airline"
    try:
        px = load_prices(["UAL", "DAL", "LUV", "XLE", "SPY"], start="2015-01-01")
        wti_raw = load_fred("DCOILWTICO", start="2014-01-01")
        wti = wti_raw.squeeze() if hasattr(wti_raw, 'squeeze') else wti_raw
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    airline_basket = (ret["UAL"] + ret["DAL"] + ret["LUV"]) / 3
    xle_r = ret["XLE"]
    spy_r = ret["SPY"]

    # Compute monthly WTI change as crack proxy
    wti_monthly = wti.resample("M").last().pct_change()

    # Compute 3-month rolling z-score of WTI monthly returns
    wti_roll_mean = wti_monthly.rolling(36).mean()
    wti_roll_std = wti_monthly.rolling(36).std()
    wti_z = (wti_monthly - wti_roll_mean) / wti_roll_std

    # Generate signal: when WTI z > 1.5 (oil spiking = jet fuel crack widening)
    # short airlines, long XLE
    signals = wti_z.dropna()
    events = []
    pair_pnl_all = pd.Series(0.0, index=ret.index)
    hold_days = 60

    for date, z_val in signals.items():
        z_float = float(z_val) if not isinstance(z_val, float) else z_val
        if np.isnan(z_float) or z_float <= 1.5:
            continue
        # Find the next trading day in our price data
        entry_mask = ret.index >= date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        # Pair PnL: long XLE - short airlines
        daily_pair = xle_r.iloc[window] - airline_basket.iloc[window]
        pair_pnl_all.iloc[window] += daily_pair.values[:exit_loc - entry_loc]

        cumulative = (1 + daily_pair).prod() - 1
        spy_cum = (1 + spy_r.iloc[window]).prod() - 1
        events.append({
            "trigger_date": str(date.date()),
            "z_score": round(float(z_val), 2),
            "entry": str(ret.index[entry_loc].date()),
            "pair_return": round(float(cumulative), 4),
            "spy_return": round(float(spy_cum), 4),
            "excess": round(float(cumulative - spy_cum), 4),
        })

    if not events:
        return mark_failed(sid, "No trigger events (WTI z > 1.5)")

    # Compute metrics on in-position days only
    in_pos = pair_pnl_all[pair_pnl_all != 0].dropna()
    if len(in_pos) < 20:
        return mark_failed(sid, f"Only {len(in_pos)} in-position days")

    m = compute_metrics(in_pos, benchmark=spy_r.reindex(in_pos.index),
                        name="Jet Fuel Crack Airline Short")
    save_result(sid, m, extra={
        "rule": "WTI 3mo z-score > 1.5 → short UAL+DAL+LUV / long XLE for 60 days",
        "mechanism": "Oil price spike widens jet fuel crack → airline margin compression on unhedged fuel exposure",
        "source": "FRED DCOILWTICO + yfinance",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
