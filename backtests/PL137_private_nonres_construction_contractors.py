"""PL137_private_nonres_construction_contractors — Private Nonres Construction YoY > +10% for 3mo → Long PWR+PRIM+ACM
When FRED PNRESCONS YoY exceeds +10% for 3 consecutive months, long contractors for 126 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL137_private_nonres_construction_contractors"
    try:
        px = load_prices(["PWR", "ACM", "SPY"], start="2002-01-01")
        pnres = load_fred("PNRESCONS", start="2000-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build basket from available tickers
    basket_tickers = [t for t in ["PWR", "ACM"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No basket tickers available")

    basket_r = ret[basket_tickers].mean(axis=1)

    # Compute YoY on monthly construction spending
    pnres_m = pnres.resample("M").last().dropna()
    pnres_yoy = pnres_m.pct_change(12)

    # Find 3-month streaks where YoY > +10%
    streak = 0
    triggers = []
    cooldown = 0
    for i in range(1, len(pnres_yoy)):
        val = float(pnres_yoy.iloc[i])
        if np.isnan(val):
            streak = 0
            continue
        if cooldown > 0:
            cooldown -= 1
            streak = 0
            continue
        if val > 0.10:
            streak += 1
            if streak == 3:
                triggers.append(pnres_yoy.index[i])
                streak = 0
                cooldown = 6
        else:
            streak = 0

    events = []
    pnl_parts = []
    hold_days = 126

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
            "pnres_yoy": round(float(pnres_yoy.loc[trig_date]), 4),
            "basket_6m_return": round(bask_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No nonres construction acceleration events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Nonres Construction Acceleration → Long Contractors")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED PNRESCONS YoY > +10% for 3 consecutive months → long PWR+ACM basket 126 days",
        "mechanism": "Sustained nonresidential construction boom (data centers, reshoring) flows to E&C contractors with 1-2 quarter lag",
        "source": "FRED PNRESCONS + yfinance",
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
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['pnres_yoy']*100:.1f}%): basket {e['basket_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
