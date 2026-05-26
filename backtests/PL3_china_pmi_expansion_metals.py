"""PL3_china_pmi_expansion_metals — China PMI Expansion Cross → Long Copper
Hand-coded PMI expansion cross events. Long HG=F for 10 trading days per event.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL3_china_pmi_expansion_metals"

    # Hand-coded PMI expansion crosses (above 50 after 2+ months below)
    events = [
        {"date": "2016-03-01", "label": "2016-03 PMI crosses 50 after contraction"},
        {"date": "2020-04-01", "label": "2020-04 PMI crosses 50 post-COVID"},
        {"date": "2023-01-02", "label": "2023-01 PMI crosses 50 post-zero-COVID"},
        {"date": "2025-04-01", "label": "2025-04 PMI crosses 50 post-tariff fears"},
    ]

    try:
        px = load_prices(["HG=F", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "HG=F" not in px.columns or px["HG=F"].dropna().shape[0] < 100:
        return mark_failed(sid, "HG=F data insufficient on yfinance")

    ret = daily_returns(px)
    copper_ret = ret["HG=F"].dropna()
    spy_ret = ret["SPY"].dropna()

    # Build daily PnL: hold copper for 10 trading days after each event date
    hold_days = 10
    pnl_series = pd.Series(0.0, index=copper_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        # Find next available trading day
        mask = copper_ret.index >= entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = copper_ret.index[mask][0]
        pos = copper_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(copper_ret))

        event_rets = copper_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        # SPY same period
        spy_mask = spy_ret.index >= entry_date
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        event_results.append({
            **ev,
            "status": "ok",
            "copper_10d_return": round(cumret, 4),
            "spy_10d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"only {len(ok_events)} valid events")

    # Compute aggregate event stats
    copper_rets_arr = np.array([e["copper_10d_return"] for e in ok_events])
    avg_ret = float(copper_rets_arr.mean())
    win_rate = float((copper_rets_arr > 0).mean())

    # Compute metrics on daily in-position PnL
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="China PMI Expansion → Long Copper")
    else:
        # Not enough in-position days for full metrics, compute simple stats
        m = {
            "name": "China PMI Expansion → Long Copper",
            "n_days": len(in_pos),
            "n_events": len(ok_events),
            "avg_event_return": round(avg_ret, 4),
            "event_win_rate": round(win_rate, 4),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "rule": "Long HG=F for 10 trading days when China Manufacturing PMI crosses above 50 after 2+ months below",
        "mechanism": "China PMI expansion cross signals industrial demand recovery; copper reprices upward",
        "source": "China NBS PMI; yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "event_win_rate": round(win_rate, 4),
    })
    print(f"Done: {len(ok_events)} events, avg 10d return={avg_ret*100:.2f}%, win rate={win_rate*100:.0f}%")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']}: copper={e['copper_10d_return']*100:.2f}%, SPY={e.get('spy_10d_return', 'N/A')}")


if __name__ == "__main__":
    main()
