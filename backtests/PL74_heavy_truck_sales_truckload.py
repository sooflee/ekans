"""PL74_heavy_truck_sales_truckload — Heavy Truck Sales Surge > +15% YoY for 3mo → Long KNX+WERN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL74_heavy_truck_sales_truckload"
    try:
        fred = load_fred("HTRUCKSSAAR", start="1990-01-01")
        ht = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if ht.empty or len(ht) < 24:
        return mark_failed(sid, "HTRUCKSSAAR data insufficient")

    yoy = ht.pct_change(12)
    yoy = yoy.dropna()

    trigger_dates = []
    streak = 0
    for i in range(len(yoy)):
        if yoy.iloc[i] > 0.15:
            streak += 1
            if streak == 3:
                if len(trigger_dates) == 0 or (yoy.index[i] - trigger_dates[-1]).days > 180:
                    trigger_dates.append(yoy.index[i])
        else:
            streak = 0

    print(f"Heavy truck sales surge events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: YoY = {yoy.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no heavy truck sales surge events found")

    try:
        px = load_prices(["KNX", "WERN", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["KNX", "WERN"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no truck tickers available")

    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 126

    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
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
            "yoy": round(float(yoy.loc[td]) * 100, 1),
            "basket_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Heavy Truck Sales → Long KNX+WERN")
    rets_arr = [e["basket_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long KNX+WERN 126 days when HTRUCKSSAAR YoY > +15% for 3 consecutive months",
        "mechanism": "Fleet expansion → freight revenue growth 2-3 quarter lag",
        "source": "FRED HTRUCKSSAAR; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
