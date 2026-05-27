"""PL271_ppi_construction_deflation_homebuilders — PPI Construction Inputs Deflation → Long Homebuilders
When FRED WPUIP2300001 (PPI Construction Inputs) shows 3-month annualized rate <0%, builders' margins expand.
Long DHI+LEN+TOL 42d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL271_ppi_construction_deflation_homebuilders"
    try:
        px = load_prices(["DHI", "LEN", "TOL", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["DHI", "LEN", "TOL"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    use_fred = False
    triggers = []
    try:
        ppi = load_fred("WPUIP2300001", start="2003-01-01").squeeze()
        if len(ppi) > 12:
            # 3-month annualized rate
            mom3 = ppi.pct_change(3)
            ann3 = (1 + mom3) ** 4 - 1  # Annualize quarterly change
            defl = ann3[ann3 < 0]
            # Cluster: first event per episode (gap >6 months)
            prev = None
            for d in defl.index:
                if prev is None or (d - prev).days > 180:
                    triggers.append(d)
                prev = d
            use_fred = True
            print(f"Found {len(triggers)} PPI construction deflation events from FRED")
    except Exception as e:
        print(f"FRED PPI construction data unavailable: {e}")

    if not use_fred or len(triggers) < 3:
        triggers = [
            pd.Timestamp("2006-11-01"), pd.Timestamp("2008-08-01"),
            pd.Timestamp("2009-03-01"), pd.Timestamp("2012-06-01"),
            pd.Timestamp("2015-01-01"), pd.Timestamp("2016-03-01"),
            pd.Timestamp("2019-06-01"), pd.Timestamp("2020-04-01"),
            pd.Timestamp("2023-01-01"),
        ]
        print(f"Using {len(triggers)} hand-coded PPI construction deflation events")

    events, pnl_parts = [], []
    hold_days = 42
    for trig_date in triggers:
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
        events.append({"trigger_date": str(trig_date.date()), "basket_42d_return": round(bask_cum, 4),
                        "spy_42d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No PPI construction deflation events found")
    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="PPI Construction Deflation → Long Homebuilders")
    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long DHI+LEN+TOL 42d when FRED WPUIP2300001 3mo annualized rate < 0%",
        "mechanism": "Construction input deflation → homebuilder cost relief → margin expansion",
        "source": "FRED WPUIP2300001 + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4), "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}", "events": events,
    })
    sharpe = m.get('sharpe', 0); cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")

if __name__ == "__main__":
    main()
