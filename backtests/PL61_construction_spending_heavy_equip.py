"""PL61_construction_spending_heavy_equip — Construction Spending YoY > +10% for 3mo → Long CAT+DE+URI
When TTLCONS YoY > +10% for 3 consecutive months: long equal-weight CAT+DE+URI for 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL61_construction_spending_heavy_equip"
    try:
        fred = load_fred("TTLCONS", start="2000-01-01")
        cons = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if cons.empty or len(cons) < 24:
        return mark_failed(sid, "TTLCONS data insufficient")

    # Compute YoY change
    yoy = cons.pct_change(12)
    yoy = yoy.dropna()

    # Find 3 consecutive months with YoY > +10%
    trigger_dates = []
    streak = 0
    for i in range(len(yoy)):
        if yoy.iloc[i] > 0.10:
            streak += 1
            if streak == 3:
                if len(trigger_dates) == 0 or (yoy.index[i] - trigger_dates[-1]).days > 180:
                    trigger_dates.append(yoy.index[i])
        else:
            streak = 0

    print(f"Construction spending surge events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: YoY = {yoy.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no construction spending surge events found")

    try:
        px = load_prices(["CAT", "DE", "URI", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["CAT", "DE", "URI"] if t in ret.columns]
    if len(available) == 0 or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

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

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "yoy_pct": round(float(yoy.loc[td]*100), 1),
            "basket_used": available,
            "basket_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Construction Spending → Long Heavy Equip")
    rets_arr = [e["basket_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long CAT+DE+URI equal-weight 126 days when TTLCONS YoY > +10% for 3 months",
        "mechanism": "Sustained construction growth → equipment demand with 1-2 quarter lag",
        "source": "FRED TTLCONS; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
