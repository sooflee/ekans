"""PL77_ism_services_expansion_xly — ISM Services PMI Crosses 55 from Below 50 → Long XLY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL77_ism_services_expansion_xly"
    try:
        fred = load_fred("NMFBAI", start="2008-01-01")
        ism = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if ism.empty or len(ism) < 10:
        return mark_failed(sid, "NMFBAI data insufficient")
    trigger_dates = []
    below50_count = 0
    fired = False
    for i in range(len(ism)):
        val = float(ism.iloc[i])
        if np.isnan(val): continue
        if val < 50:
            below50_count += 1
            fired = False
        elif val >= 55 and below50_count >= 3 and not fired:
            trigger_dates.append(ism.index[i])
            fired = True
            below50_count = 0
        elif val >= 50:
            pass  # between 50 and 55, keep counting
    print(f"ISM Services expansion events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: ISM = {ism.loc[d]:.1f}")
    if not trigger_dates:
        return mark_failed(sid, "no ISM Services expansion events found")
    try:
        px = load_prices(["XLY", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    xly_ret = ret["XLY"]; spy_ret = ret["SPY"]
    hold_days = 126
    pnl_series = pd.Series(0.0, index=xly_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = xly_ret.index >= entry_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = xly_ret.index[entry_mask][0]
        pos = xly_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(xly_ret))
        event_rets = xly_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "ism_value": round(float(ism.loc[td]), 1),
            "xly_6m_return": round(cumret, 4), "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="ISM Services Expansion → Long XLY")
    rets_arr = [e["xly_6m_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long XLY 126d when NMFBAI crosses 55 after 3+ months below 50",
        "source": "FRED NMFBAI; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
