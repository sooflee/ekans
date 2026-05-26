"""PL145_dgorder_acceleration_semi_equipment — Durable Goods 3mo MoM Acceleration → Long AMAT+LRCX+KLAC 63d
When FRED DGORDER MoM is positive AND increasing for 3 consecutive months, long semi equipment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL145_dgorder_acceleration_semi_equipment"
    try:
        px = load_prices(["AMAT", "LRCX", "KLAC", "SPY"], start="1995-01-01")
        dgo = load_fred("DGORDER", start="1992-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["AMAT", "LRCX", "KLAC"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No semi equipment tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly durable goods, compute MoM
    dgo_m = dgo.resample("M").last().dropna()
    dgo_mom = dgo_m.pct_change()

    # Find 3-month acceleration streaks (positive AND increasing MoM)
    triggers = []
    cooldown = 0
    for i in range(3, len(dgo_mom)):
        if cooldown > 0:
            cooldown -= 1
            continue
        m0 = float(dgo_mom.iloc[i-2])
        m1 = float(dgo_mom.iloc[i-1])
        m2 = float(dgo_mom.iloc[i])
        if np.isnan(m0) or np.isnan(m1) or np.isnan(m2):
            continue
        if m0 > 0 and m1 > m0 and m2 > m1:
            triggers.append(dgo_mom.index[i])
            cooldown = 3

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

        events.append({
            "trigger_date": str(trig_date.date()),
            "dgo_mom": round(float(dgo_mom.loc[trig_date]) * 100, 2),
            "basket_63d_return": round(bask_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No durable goods acceleration events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Durable Goods Acceleration → Long Semi Equipment")

    avg_basket = np.mean([e["basket_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED DGORDER 3mo positive and increasing MoM → long AMAT+LRCX+KLAC 63 days",
        "mechanism": "Durable goods acceleration signals capex cycle turn — semi equipment benefits disproportionately from manufacturing capex",
        "source": "FRED DGORDER + yfinance",
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
        print(f"  {flag} {e['trigger_date']} (MoM={e['dgo_mom']:.1f}%): basket {e['basket_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
