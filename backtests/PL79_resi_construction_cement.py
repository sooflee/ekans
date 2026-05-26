"""PL79_resi_construction_cement — Private Resi Construction 3mo Ann. > +15% → Long EXP+SUM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL79_resi_construction_cement"
    try:
        fred = load_fred("PRRESCONS", start="2000-01-01")
        data = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if data.empty or len(data) < 6:
        return mark_failed(sid, "PRRESCONS data insufficient")
    # 3-month annualized growth = (P_t / P_{t-3})^4 - 1
    ann3m = (data / data.shift(3)) ** 4 - 1
    ann3m = ann3m.dropna()
    trigger_dates = []
    prev_above = False
    for i in range(len(ann3m)):
        val = float(ann3m.iloc[i])
        if np.isnan(val): continue
        if val > 0.15 and not prev_above:
            if len(trigger_dates) == 0 or (ann3m.index[i] - trigger_dates[-1]).days > 180:
                trigger_dates.append(ann3m.index[i])
            prev_above = True
        elif val <= 0.15:
            prev_above = False
    print(f"Resi construction surge events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no events found")
    try:
        px = load_prices(["EXP", "SUM", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EXP", "SUM"] if t in ret.columns]
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
        event_results.append({"trigger_date": str(td.date()), "ann3m": round(float(ann3m.loc[td])*100, 1),
            "basket_return": round(cumret, 4), "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Resi Construction → Long EXP+SUM")
    rets_arr = [e["basket_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long EXP+SUM 126d when PRRESCONS 3mo ann. > +15%",
        "source": "FRED PRRESCONS; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
