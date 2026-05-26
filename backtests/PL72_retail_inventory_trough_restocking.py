"""PL72_retail_inventory_trough_restocking — Retail Inventory/Sales Ratio Trough → Long WMT+TGT+COST
Find local minima in FRED RETAILIRSA. Long WMT+TGT+COST 126 days from month after trough.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL72_retail_inventory_trough_restocking"
    try:
        px = load_prices(["WMT", "TGT", "COST", "SPY"], start="2000-01-01")
        inv = load_fred("RETAILIRSA", start="1992-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Monthly inventory/sales ratio
    inv_m = inv.resample("ME").last().dropna()

    # Find local minima (lower than both prior and next month)
    # Also require the trough to be meaningful — declining trend into it
    triggers = []
    last_trigger = None
    for i in range(2, len(inv_m) - 1):
        prev2 = float(inv_m.iloc[i - 2])
        prev = float(inv_m.iloc[i - 1])
        curr = float(inv_m.iloc[i])
        nxt = float(inv_m.iloc[i + 1])
        if np.isnan(prev) or np.isnan(curr) or np.isnan(nxt) or np.isnan(prev2):
            continue
        # Local minimum: lower than prior and next
        if curr < prev and curr < nxt:
            # Additional filter: declining into trough (prev2 > prev > curr) to avoid noise
            if prev2 > prev or prev > curr:
                # Entry is the month after the trough (so we know it's a trough)
                trig = inv_m.index[i + 1]
                if last_trigger is None or (trig - last_trigger).days > 180:
                    triggers.append(trig)
                    last_trigger = trig

    print(f"Found {len(triggers)} retail inventory trough events")

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

        # Equal-weight basket of available tickers
        basket_tickers = [t for t in ["WMT", "TGT", "COST"] if t in ret.columns
                          and not ret[t].iloc[window].isna().all()]
        if not basket_tickers:
            continue

        basket_r = ret[basket_tickers].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        # Get inventory value at trough (month before trigger)
        trough_idx = inv_m.index.get_indexer([trig_date], method='pad')[0] - 1
        inv_val = float(inv_m.iloc[max(0, trough_idx)]) if trough_idx >= 0 else np.nan

        events.append({
            "trigger_date": str(trig_date.date()),
            "inv_ratio_at_trough": round(inv_val, 4) if not np.isnan(inv_val) else None,
            "basket_tickers": basket_tickers,
            "basket_6m_return": round(basket_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No retail inventory trough events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Retail Inventory Trough → Long Retailers")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED RETAILIRSA local minimum → long WMT+TGT+COST 6mo from month after trough",
        "mechanism": "Lean shelves → restocking orders → supply chain revenue boost for major retailers",
        "source": "FRED RETAILIRSA + yfinance",
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
    for e in events[:10]:
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        inv_str = f"inv={e['inv_ratio_at_trough']}" if e.get('inv_ratio_at_trough') else ""
        print(f"  {flag} {e['trigger_date']} ({inv_str}): basket {e['basket_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if len(events) > 10:
        print(f"  ... and {len(events)-10} more events")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
