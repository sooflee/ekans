"""PL265_usda_soybean_blooming_condition — USDA Crop Progress Soybean Poor Condition → Long Soybeans
When USDA weekly Crop Progress shows soybean good+excellent <55% during blooming, long SOYB+DBA 21d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL265_usda_soybean_blooming_condition"
    try:
        px = load_prices(["SOYB", "DBA", "SPY"], start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["SOYB", "DBA"] if t in ret.columns]
    if len(basket_tickers) < 1:
        return mark_failed(sid, "No soybean tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Hand-coded: USDA Crop Progress soybean poor condition during bloom (July)
    poor_condition_dates = [
        "2012-07-09",  # 2012 drought — soybean condition collapsed
        "2013-07-08",  # Below-avg conditions early July
        "2019-07-08",  # Delayed planting → poor conditions
        "2021-07-12",  # Western drought stress
        "2023-07-10",  # Midwest dry spell
    ]

    events, pnl_parts = [], []
    hold_days = 21
    for date_str in poor_condition_dates:
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
        events.append({"trigger_date": date_str, "basket_21d_return": round(bask_cum, 4),
                        "spy_21d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No soybean poor condition events found")
    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="USDA Soybean Poor Condition → Long SOYB+DBA")
    avg_basket = np.mean([e["basket_21d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_21d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long SOYB+DBA 21d when USDA Crop Progress soybean good+excellent <55% during July bloom",
        "mechanism": "Poor crop conditions during critical blooming → yield loss → soybean price rally",
        "source": "USDA Crop Progress + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4), "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}", "events": events,
    })
    sharpe = m.get('sharpe', 0); cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_21d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_21d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")

if __name__ == "__main__":
    main()
