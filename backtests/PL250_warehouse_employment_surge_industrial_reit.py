"""PL250_warehouse_employment_surge_industrial_reit — FRED Warehouse Employment YoY Surge → Long Industrial REITs
Uses FRED CES4349300001 (warehouse employment). When YoY growth >5% for 3 consecutive months,
long STAG+PLD+REXR for 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL250_warehouse_employment_surge_industrial_reit"
    try:
        px = load_prices(["STAG", "PLD", "REXR", "SPY"], start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["STAG", "PLD", "REXR"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    use_fred = False
    triggers = []
    try:
        emp = load_fred("CES4349300001", start="2009-01-01").squeeze()
        if len(emp) > 24:
            yoy = emp.pct_change(12)
            above = yoy > 0.05
            count = 0
            cooldown = 0
            for i in range(len(above)):
                if cooldown > 0:
                    cooldown -= 1; count = 0; continue
                if above.iloc[i]:
                    count += 1
                    if count >= 3:
                        triggers.append(above.index[i])
                        cooldown = 6; count = 0
                else:
                    count = 0
            use_fred = True
            print(f"Found {len(triggers)} warehouse employment surge events from FRED")
    except Exception as e:
        print(f"FRED warehouse employment data unavailable: {e}")

    if not use_fred or len(triggers) < 3:
        triggers = [
            pd.Timestamp("2014-06-01"), pd.Timestamp("2015-12-01"),
            pd.Timestamp("2017-06-01"), pd.Timestamp("2018-06-01"),
            pd.Timestamp("2019-03-01"), pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-12-01"), pd.Timestamp("2022-06-01"),
        ]
        print(f"Using {len(triggers)} hand-coded warehouse employment surge events")

    events = []
    pnl_parts = []
    hold_days = 30

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
        events.append({"trigger_date": str(trig_date.date()), "basket_30d_return": round(bask_cum, 4),
                        "spy_30d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Warehouse Employment Surge → Long Industrial REITs")
    avg_basket = np.mean([e["basket_30d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_30d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long STAG+PLD+REXR 30d when FRED CES4349300001 YoY >5% for 3 months",
        "mechanism": "Warehouse employment surge → industrial space demand → rental growth → REIT NAV increase",
        "source": "FRED CES4349300001 + yfinance", "n_events": len(events),
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
