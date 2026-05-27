"""PL489_doe_iac_energy_efficiency_equipment
DOE Industrial Assessment Center Recommendation Surge -> Energy Efficiency Equipment -> ROK/EMR Long
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL489_doe_iac_energy_efficiency_equipment"

    # DOE IAC publishes annual data. Hand-coded years where implementation
    # rate increased AND savings per assessment rose significantly.
    events = [
        {"date": "2008-06-01", "label": "2008: High energy prices drove IAC adoption surge", "year": 2008},
        {"date": "2010-03-01", "label": "2010: Post-ARRA efficiency stimulus", "year": 2010},
        {"date": "2014-03-01", "label": "2014: Manufacturing efficiency push", "year": 2014},
        {"date": "2017-06-01", "label": "2017: Industrial IoT driving efficiency investment", "year": 2017},
        {"date": "2022-06-01", "label": "2022: Energy price spike drove efficiency retrofits", "year": 2022},
        {"date": "2023-06-01", "label": "2023: IRA incentives boosted industrial efficiency", "year": 2023},
    ]

    try:
        px = load_prices(["ROK", "EMR", "SPY"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    ret = daily_returns(px)
    combo_ret = ret[["ROK", "EMR"]].mean(axis=1).dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 40
    pnl_series = pd.Series(0.0, index=combo_ret.index)
    positions = pd.Series(0.0, index=combo_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        mask = combo_ret.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = combo_ret.index[mask][0]
        pos = combo_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(combo_ret))

        event_rets = combo_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        spy_mask = spy_ret.index >= start_idx
        spy_cumret = None
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        for i in range(pos, end_pos):
            idx = combo_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = combo_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({**ev, "status": "ok",
            "portfolio_40d_return": round(cumret, 4),
            "spy_40d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None})

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    rets_arr = np.array([e["portfolio_40d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(rets_arr.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((rets_arr > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="DOE IAC -> ROK+EMR Long", positions=positions[positions != 0])
    else:
        m = {"name": "DOE IAC -> ROK+EMR Long", "n_days": len(in_pos),
             "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
             "cagr": avg_ret,
             "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
             "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0}

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When DOE IAC shows rising efficiency adoption + savings, long ROK+EMR 40d",
        "mechanism": "Industrial efficiency investment surge -> automation equipment demand -> ROK/EMR revenue",
        "source": "DOE IAC + yfinance",
        "events": event_results, "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg={avg_ret*100:.2f}%, win={win_rate*100:.0f}%, sharpe={m.get('sharpe','N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']}: {e['portfolio_40d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")

if __name__ == "__main__":
    main()
