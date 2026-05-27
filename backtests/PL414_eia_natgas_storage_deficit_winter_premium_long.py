"""PL414_eia_natgas_storage_deficit_winter_premium_long — EIA Storage Below 5yr Avg Mid-Injection -> Long Nat Gas E&Ps
When FRED NGTTSTUS1W (NG working storage) is > 10% below 5-year same-week average
during injection season (Apr-Oct), long AR+EQT for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL414_eia_natgas_storage_deficit_winter_premium_long"
    try:
        px = load_prices(["AR", "EQT", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load EIA natural gas storage from FRED
    storage = None
    for series in ["NGTTSTUS1W", "NG_STORAGE_TOTAL"]:
        try:
            storage = load_fred(series, start="2005-01-01").squeeze()
            if storage.dropna().empty:
                storage = None
                continue
            break
        except Exception:
            continue
    if storage is None:
        return mark_failed(sid, "Could not load FRED natural gas storage (NGTTSTUS1W)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["AR", "EQT"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No natgas producer tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Compute 5-year same-week average
    storage = storage.dropna()
    storage.index = pd.to_datetime(storage.index)
    storage_wk = storage.resample("W").last().dropna()

    # For each week, compute 5-year average for that week-of-year
    storage_wk_num = storage_wk.index.isocalendar().week.values
    five_yr_avg = pd.Series(index=storage_wk.index, dtype=float)

    for i in range(260, len(storage_wk)):  # start after 5 years
        wk = storage_wk_num[i]
        # Get same-week values from past 5 years
        past_mask = (storage_wk_num[:i] == wk)
        past_vals = storage_wk.iloc[:i][past_mask].tail(5)
        if len(past_vals) >= 3:
            five_yr_avg.iloc[i] = past_vals.mean()

    # Find injection season (Apr-Oct) deficits > 10%
    deficit = (storage_wk - five_yr_avg) / five_yr_avg
    triggers = []
    last_trigger = None

    for i in range(len(deficit)):
        dt = deficit.index[i]
        val = float(deficit.iloc[i])
        if np.isnan(val):
            continue
        if dt.month not in (4, 5, 6, 7, 8, 9, 10):
            continue
        if val < -0.10:
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No storage deficit > 10% events during injection season")

    events = []
    pnl_parts = []
    hold_days = 42

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "storage_deficit_pct": round(float(deficit.loc[trig_date]) * 100, 1),
            "basket_42d_return": round(bask_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable storage deficit events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="EIA NG Storage Deficit -> Long AR+EQT")

    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "EIA NG working storage > 10% below 5yr avg during Apr-Oct injection season -> long AR+EQT 42d",
        "mechanism": "Storage deficit during injection season implies winter premium building, benefiting natgas producers",
        "source": "FRED NGTTSTUS1W + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: deficit={e['storage_deficit_pct']:.0f}%, basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
