"""PL121_import_surge_warehouse_reits -- Real Goods Import Surge -> Long PLD+STAG
Long PLD+STAG 126d when IMPGS YoY > +8% for 3 months
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL121_import_surge_warehouse_reits"
    
    # Load FRED data
    try:
        fred = load_fred(['IMPGS'], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED data empty")
    
    series = fred["IMPGS"].dropna()
    if len(series) < 13:
        return mark_failed(sid, "insufficient data for YoY")
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    above = yoy > 8
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(above)):
        if above.iloc[i]:
            streak += 1
            if streak >= 3 and not fired:
                trigger_dates.append(yoy.index[i])
                fired = True
        else:
            streak = 0
            fired = False

    
    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")
    
    print(f"Trigger events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}")
    
    # Load prices
    try:
        px = load_prices(['PLD', 'STAG', 'SPY'], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    
    ret = daily_returns(px)
    
    # Build basket
    trade_tickers = ['PLD', 'STAG']
    available = [t for t in trade_tickers if t in ret.columns]
    if len(available) == 0:
        return mark_failed(sid, f"no trade tickers available in data")
    
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"] if "SPY" in ret.columns else basket_ret * 0
    
    hold_days = 126
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1) if not isinstance(td, pd.Timestamp) else td
        mask = basket_ret.index >= entry_date
        if mask.sum() < max(30, hold_days // 2):
            continue
        entry_idx = basket_ret.index[mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        if end_pos - pos < 30:
            continue
        
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        
        spy_cumret = None
        spy_start = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        if spy_start is not None:
            se = min(spy_start + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[spy_start:se]).prod() - 1)
        
        event_results.append({
            "trigger_date": str(td.date() if hasattr(td, 'date') else td),
            "return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })
    
    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after price alignment")
    
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")
    
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Real Goods Import Surge -> Long PLD+STAG")
    rets_arr = [e["return"] for e in event_results]
    
    save_result(sid, m, extra={
        "rule": "Long PLD+STAG 126d when IMPGS YoY > +8% for 3 months",
        "source": "FRED IMPGS; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
