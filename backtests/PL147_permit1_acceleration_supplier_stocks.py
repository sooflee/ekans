"""PL147_permit1_acceleration_supplier_stocks — PERMIT1 YoY Improving 3 Months → Long SHW+VMC+MLM+BLDR 63d
When FRED PERMIT1 YoY rate improves for 3 consecutive months, long building supply stocks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL147_permit1_acceleration_supplier_stocks"
    try:
        px = load_prices(["SHW", "VMC", "MLM", "SPY"], start="1994-01-01")
        permit = load_fred("PERMIT1", start="1990-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["SHW", "VMC", "MLM"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No building supply tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly permits, compute YoY
    perm_m = permit.resample("M").last().dropna()
    perm_yoy = perm_m.pct_change(12)
    # Second derivative: month-over-month change in YoY
    perm_yoy_delta = perm_yoy.diff()

    # Find 3 consecutive months of positive second derivative (accelerating YoY)
    triggers = []
    cooldown = 0
    streak = 0
    for i in range(1, len(perm_yoy_delta)):
        val = float(perm_yoy_delta.iloc[i])
        if np.isnan(val):
            streak = 0
            continue
        if cooldown > 0:
            cooldown -= 1
            streak = 0
            continue
        if val > 0:
            streak += 1
            if streak == 3:
                triggers.append(perm_yoy_delta.index[i])
                streak = 0
                cooldown = 3
        else:
            streak = 0

    events = []
    pnl_parts = []
    hold_days = 63

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

        # Get the YoY at trigger
        yoy_val = float(perm_yoy.loc[trig_date]) if trig_date in perm_yoy.index else np.nan

        events.append({
            "trigger_date": str(trig_date.date()),
            "permit1_yoy": round(yoy_val * 100, 1) if not np.isnan(yoy_val) else None,
            "basket_63d_return": round(bask_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No permit acceleration events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="PERMIT1 Acceleration → Long Suppliers")

    avg_basket = np.mean([e["basket_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED PERMIT1 YoY improving for 3 consecutive months → long SHW+VMC+MLM 63 days",
        "mechanism": "Single-family permit acceleration signals housing trough — building material suppliers benefit first in the cycle",
        "source": "FRED PERMIT1 + yfinance",
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
        flag = "+" if e["basket_63d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['permit1_yoy']}%): basket {e['basket_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
