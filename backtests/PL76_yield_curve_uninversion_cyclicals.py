"""PL76_yield_curve_uninversion_cyclicals — 10Y-2Y Un-Inversion After 12mo → Long XLI+XLF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL76_yield_curve_uninversion_cyclicals"
    try:
        fred = load_fred("T10Y2Y", start="1976-01-01")
        spread = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if spread.empty:
        return mark_failed(sid, "T10Y2Y data empty")
    # Find dates where spread crosses above 0 after 12+ months below 0
    trigger_dates = []
    below_zero_days = 0
    for i in range(1, len(spread)):
        val = float(spread.iloc[i])
        prev = float(spread.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        if val < 0:
            below_zero_days += 1
        elif val >= 0 and prev < 0 and below_zero_days >= 252:
            if len(trigger_dates) == 0 or (spread.index[i] - trigger_dates[-1]).days > 365:
                trigger_dates.append(spread.index[i])
            below_zero_days = 0
        elif val >= 0:
            below_zero_days = 0
    print(f"Yield curve un-inversion events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}")
    if not trigger_dates:
        return mark_failed(sid, "no un-inversion events found")
    try:
        px = load_prices(["XLI", "XLF", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["XLI", "XLF"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 252
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_mask = basket_ret.index >= td
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
        event_results.append({"trigger_date": str(td.date()), "basket_12m_return": round(cumret, 4),
                              "spy_12m_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events after alignment")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Yield Curve Un-Inversion → Long XLI+XLF")
    rets_arr = [e["basket_12m_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long XLI+XLF 252d when T10Y2Y crosses above 0 after 12+ months inverted",
        "mechanism": "Un-inversion signals recession troughing → cyclical recovery",
        "source": "FRED T10Y2Y; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
