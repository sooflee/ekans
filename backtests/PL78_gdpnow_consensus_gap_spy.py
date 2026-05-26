"""PL78_gdpnow_consensus_gap_spy — GDPNow > 3.0% → Long SPY 42d"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL78_gdpnow_consensus_gap_spy"
    try:
        fred = load_fred("GDPNOW", start="2011-01-01")
        gdp = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if gdp.empty or len(gdp) < 10:
        return mark_failed(sid, "GDPNOW data insufficient")
    trigger_dates = []
    prev_above = False
    for i in range(len(gdp)):
        val = float(gdp.iloc[i])
        if np.isnan(val): continue
        if val > 3.0 and not prev_above:
            if len(trigger_dates) == 0 or (gdp.index[i] - trigger_dates[-1]).days > 60:
                trigger_dates.append(gdp.index[i])
            prev_above = True
        elif val <= 3.0:
            prev_above = False
    print(f"GDPNow > 3.0% events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no GDPNow events found")
    try:
        px = load_prices("SPY", start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    if isinstance(ret, pd.DataFrame): spy_ret = ret.iloc[:, 0]
    else: spy_ret = ret
    hold_days = 42
    pnl_series = pd.Series(0.0, index=spy_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_mask = spy_ret.index >= td
        if entry_mask.sum() < hold_days: continue
        entry_idx = spy_ret.index[entry_mask][0]
        pos = spy_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(spy_ret))
        event_rets = spy_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        event_results.append({"trigger_date": str(td.date()), "gdpnow": round(float(gdp.loc[td]), 2),
            "spy_return": round(cumret, 4)})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="GDPNow > 3% → Long SPY")
    rets_arr = [e["spy_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long SPY 42d when GDPNOW > 3.0%",
        "source": "FRED GDPNOW; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
