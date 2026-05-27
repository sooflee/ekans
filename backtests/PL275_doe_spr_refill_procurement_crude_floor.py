"""PL275_doe_spr_refill_procurement_crude_floor — DOE SPR Refill → Long USO+XOP
When DOE announces SPR refill procurement at target price, establishes crude floor → long USO+XOP 30d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL275_doe_spr_refill_procurement_crude_floor"
    try:
        px = load_prices(["USO", "XOP", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["USO", "XOP"] if t in ret.columns]
    if len(basket_tickers) < 1:
        return mark_failed(sid, "No crude/energy tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # DOE SPR refill/procurement announcement dates (public)
    spr_dates = [
        "2023-01-12",  # DOE announced SPR refill plan at ~$70/bbl
        "2023-06-14",  # DOE awarded 3M barrels for SPR refill
        "2023-10-04",  # DOE purchased 6M barrels for SPR
        "2024-01-31",  # DOE continued SPR purchases
        "2024-05-16",  # DOE SPR refill purchase at ~$79
        "2024-08-20",  # DOE awarded additional SPR barrels
    ]
    events, pnl_parts = [], []
    hold_days = 30
    for date_str in spr_dates:
        trig_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)
        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        events.append({"trigger_date": date_str, "basket_30d_return": round(bask_cum, 4),
                        "spy_30d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No SPR refill events found")
    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="DOE SPR Refill → Long Crude/E&P")
    avg_basket = np.mean([e["basket_30d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_30d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long USO+XOP 30d when DOE announces SPR refill procurement",
        "mechanism": "DOE SPR refill → establishes crude price floor → E&P companies benefit from price support",
        "source": "DOE/SPR announcements + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4), "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}", "events": events,
    })
    sharpe = m.get('sharpe', 0); cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_30d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_30d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")

if __name__ == "__main__":
    main()
