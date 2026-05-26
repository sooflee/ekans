"""PL80_northeast_permits_building_products — NE Permits >+20% YoY (National Flat) → Long JELD+AZEK+BLDR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL80_northeast_permits_building_products"
    try:
        ne = load_fred("PERMITNE", start="2000-01-01").squeeze()
        nat = load_fred("PERMIT", start="2000-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if ne.empty or nat.empty:
        return mark_failed(sid, "permit data insufficient")
    ne_yoy = ne.pct_change(12).dropna()
    nat_yoy = nat.pct_change(12).dropna()
    idx = ne_yoy.index.intersection(nat_yoy.index)
    trigger_dates = []
    for d in idx:
        ne_v = float(ne_yoy.loc[d]); nat_v = float(nat_yoy.loc[d])
        if np.isnan(ne_v) or np.isnan(nat_v): continue
        if ne_v > 0.20 and nat_v < 0.05:
            if len(trigger_dates) == 0 or (d - trigger_dates[-1]).days > 180:
                trigger_dates.append(d)
    print(f"NE permit outperformance events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no events found")
    try:
        px = load_prices(["JELD", "AZEK", "BLDR", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["JELD", "AZEK", "BLDR"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 126
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "basket_return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events after alignment (tickers too recent)")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="NE Permits → Long Building Products")
    rets_arr = [e["basket_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long JELD+AZEK+BLDR 126d when PERMITNE YoY>20% and PERMIT YoY<5%",
        "source": "FRED PERMITNE, PERMIT; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
