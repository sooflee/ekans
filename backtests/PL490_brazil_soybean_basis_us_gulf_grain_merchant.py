"""PL490_brazil_soybean_basis_us_gulf_grain_merchant
Brazil Soybean Export Basis Widening -> US Gulf demand shift -> ADM/BG Long
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL490_brazil_soybean_basis_us_gulf_grain_merchant"

    # Brazil soybean basis data is not on FRED. Use hand-coded events where
    # Brazil logistics disruptions shifted demand to US Gulf.
    events = [
        {"date": "2014-02-10", "label": "2014: Brazil port congestion during record harvest", "months": "Jan-Feb 2014"},
        {"date": "2015-11-02", "label": "2015: Brazil trucking cost surge + Santos congestion", "months": "Oct-Nov 2015"},
        {"date": "2018-05-28", "label": "2018: Brazil trucker strike shut Paranagua/Santos 10+ days", "months": "May 2018"},
        {"date": "2020-03-02", "label": "2020: Record Brazil soy crop caused port congestion", "months": "Feb-Mar 2020"},
        {"date": "2021-03-01", "label": "2021: La Nina delayed Brazil harvest + logistics constraints", "months": "Feb-Mar 2021"},
        {"date": "2023-03-06", "label": "2023: Paranagua congestion during bumper crop", "months": "Feb-Mar 2023"},
        {"date": "2024-02-05", "label": "2024: Northern arc logistic bottlenecks at new ports", "months": "Jan-Feb 2024"},
    ]

    try:
        px = load_prices(["ADM", "BG", "SPY"], start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    ret = daily_returns(px)
    grain_ret = ret[["ADM", "BG"]].mean(axis=1).dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 20
    min_gap = 90
    pnl_series = pd.Series(0.0, index=grain_ret.index)
    positions = pd.Series(0.0, index=grain_ret.index)
    event_results = []
    last_signal = None

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        if last_signal is not None and (entry_date - last_signal).days < min_gap:
            event_results.append({**ev, "status": "skipped", "reason": f"too close ({(entry_date - last_signal).days}d)"})
            continue

        mask = grain_ret.index >= entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = grain_ret.index[mask][0]
        pos = grain_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(grain_ret))

        event_rets = grain_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        spy_mask = spy_ret.index >= start_idx
        spy_cumret = None
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        for i in range(pos, end_pos):
            idx = grain_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = grain_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({**ev, "status": "ok",
            "grain_20d_return": round(cumret, 4),
            "spy_20d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None})
        last_signal = entry_date

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    rets_arr = np.array([e["grain_20d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(rets_arr.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((rets_arr > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Brazil Basis -> ADM+BG Long", positions=positions[positions != 0])
    else:
        m = {"name": "Brazil Basis -> ADM+BG Long", "n_days": len(in_pos),
             "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
             "cagr": avg_ret,
             "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
             "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0}

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When Brazil soy logistics disruption widens basis favoring US Gulf, long ADM+BG 20d",
        "mechanism": "Brazil logistics disruption -> demand shift to US Gulf -> grain merchant origination margins surge",
        "source": "Brazil port/logistics events + yfinance",
        "events": event_results, "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg={avg_ret*100:.2f}%, win={win_rate*100:.0f}%, sharpe={m.get('sharpe','N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']}: ADM+BG={e['grain_20d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")

if __name__ == "__main__":
    main()
