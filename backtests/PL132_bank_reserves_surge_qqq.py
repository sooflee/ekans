"""PL132_bank_reserves_surge_qqq -- Bank Reserves Surge -> Long QQQ
Long QQQ 126d when TOTRESNS 3-month rolling change > +15%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL132_bank_reserves_surge_qqq"
    try:
        fred = load_fred("TOTRESNS", start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED data empty")
    
    series = fred.squeeze().dropna()
    if len(series) < 7:
        return mark_failed(sid, "insufficient data")
    
    # 3-month rolling pct change
    pct3m = series.pct_change(6) * 100  # biweekly data, ~6 obs = 3 months
    pct3m = pct3m.dropna()
    
    trigger_dates = []
    fired = False
    for i in range(len(pct3m)):
        if pct3m.iloc[i] > 15 and not fired:
            trigger_dates.append(pct3m.index[i])
            fired = True
        elif pct3m.iloc[i] < 5:
            fired = False
    
    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")
    
    print(f"Trigger events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}")
    
    try:
        px = load_prices(["QQQ", "SPY"], start="1999-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    
    ret = daily_returns(px)
    if "QQQ" not in ret.columns:
        return mark_failed(sid, "QQQ not available")
    
    basket_ret = ret["QQQ"]
    spy_ret = ret["SPY"] if "SPY" in ret.columns else basket_ret * 0
    
    hold_days = 126
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        mask = basket_ret.index >= entry_date
        if mask.sum() < 30:
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
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        
        event_results.append({
            "trigger_date": str(td.date()),
            "return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })
    
    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after price alignment")
    
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")
    
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Bank Reserves Surge -> Long QQQ")
    rets_arr = [e["return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long QQQ 126d when TOTRESNS 3-month rolling change > +15%",
        "source": "FRED TOTRESNS; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
