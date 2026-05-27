"""PL409_census_m3_core_capex_industrial_dist_long — Census M3 Core Capex Shipments Acceleration -> Long Industrial Distributors
When FRED A34SNO (nondefense capex ex-aircraft) YoY turns positive after 3+ negative months,
or accelerates > 3pp from trough, long FAST+GWW for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL409_census_m3_core_capex_industrial_dist_long"
    try:
        px = load_prices(["FAST", "GWW", "SPY"], start="1992-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load Census M3 core capex from FRED
    capex = None
    for series in ["A34SNO", "ANXFS", "NEWORDER"]:
        try:
            capex = load_fred(series, start="1992-01-01").squeeze()
            if capex.dropna().empty:
                capex = None
                continue
            break
        except Exception:
            continue
    if capex is None:
        return mark_failed(sid, "Could not load FRED core capex (A34SNO / ANXFS)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["FAST", "GWW"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No industrial distributor tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly capex, YoY growth
    capex_m = capex.resample("M").last().dropna()
    capex_yoy = capex_m.pct_change(12)

    # Find inflection points:
    # 1) YoY turns positive after 3+ negative months
    # 2) Or accelerates > 3pp from trough
    triggers = []
    last_trigger = None
    neg_streak = 0
    trough_yoy = 0.0

    for i in range(12, len(capex_yoy)):
        dt = capex_yoy.index[i]
        val = float(capex_yoy.iloc[i])
        prev = float(capex_yoy.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            neg_streak = 0
            continue

        if val < 0:
            neg_streak += 1
            trough_yoy = min(trough_yoy, val)
        else:
            # Check condition 1: turning positive after 3+ negative
            if neg_streak >= 3 and prev < 0:
                if last_trigger is None or (dt - last_trigger).days >= 90:
                    triggers.append(dt)
                    last_trigger = dt
            neg_streak = 0
            trough_yoy = 0.0

        # Check condition 2: acceleration > 3pp from trough
        if neg_streak > 0 and val - trough_yoy > 0.03:
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No capex inflection events found")

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
            "capex_yoy": round(float(capex_yoy.loc[trig_date]), 4),
            "basket_42d_return": round(bask_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable capex inflection events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Census M3 Core Capex Inflection -> Long FAST+GWW")

    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED A34SNO YoY turns positive after 3+ negative months -> long FAST+GWW 42 days",
        "mechanism": "Core capex shipment inflection drives downstream MRO consumable demand recovery at industrial distributors",
        "source": "FRED A34SNO + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: capex_yoy={e['capex_yoy']*100:.1f}%, basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
