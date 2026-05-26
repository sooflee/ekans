"""PL6_cdc_wastewater_viral_pharma — CDC Wastewater Viral Surge → Pharma
Hand-coded viral surge events. Long equal-weight DGX+LH+MRNA for 30 trading days per event.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL6_cdc_wastewater_viral_pharma"

    # Hand-coded viral surge events
    events = [
        {"date": "2021-12-01", "label": "Omicron wave"},
        {"date": "2022-12-01", "label": "Winter tripledemic (flu/RSV/COVID)"},
        {"date": "2023-09-01", "label": "Fall COVID wave"},
        {"date": "2024-12-01", "label": "Winter RSV/flu wave"},
    ]

    try:
        px = load_prices(["DGX", "LH", "MRNA", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["DGX", "LH", "MRNA", "SPY"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} data insufficient")

    ret = daily_returns(px)
    # Pharma basket: equal-weight DGX+LH+MRNA
    basket_ret = (ret["DGX"] + ret["LH"] + ret["MRNA"]) / 3
    spy_ret = ret["SPY"]

    hold_days = 30
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient data"})
            continue

        start_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(basket_ret))

        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        # Per-component returns
        dgx_cumret = float((1 + ret["DGX"].iloc[pos:end_pos]).prod() - 1)
        lh_cumret = float((1 + ret["LH"].iloc[pos:end_pos]).prod() - 1)
        mrna_cumret = float((1 + ret["MRNA"].iloc[pos:end_pos]).prod() - 1)

        # SPY same period
        spy_mask2 = spy_ret.index >= start_idx
        if spy_mask2.sum() >= hold_days:
            spy_start = spy_ret.index.get_loc(spy_ret.index[spy_mask2][0])
            spy_event = spy_ret.iloc[spy_start:spy_start + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        event_results.append({
            "date": ev["date"],
            "label": ev["label"],
            "status": "ok",
            "basket_30d_return": round(cumret, 4),
            "DGX_30d_return": round(dgx_cumret, 4),
            "LH_30d_return": round(lh_cumret, 4),
            "MRNA_30d_return": round(mrna_cumret, 4),
            "SPY_30d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_vs_SPY": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e.get("status") == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"only {len(ok_events)} valid events")

    basket_rets_arr = np.array([e["basket_30d_return"] for e in ok_events])
    avg_ret = float(basket_rets_arr.mean())
    win_rate = float((basket_rets_arr > 0).mean())

    excess_rets = np.array([e["excess_vs_SPY"] for e in ok_events if e.get("excess_vs_SPY") is not None])
    avg_excess = float(excess_rets.mean()) if len(excess_rets) > 0 else None

    # Compute metrics on in-position days
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Viral Surge → Long Pharma Basket")
    else:
        m = {
            "name": "Viral Surge → Long Pharma Basket",
            "n_days": len(in_pos),
            "n_events": len(ok_events),
            "avg_event_return": round(avg_ret, 4),
            "event_win_rate": round(win_rate, 4),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "rule": "Long equal-weight DGX+LH+MRNA for 30 trading days at start of viral surge events",
        "mechanism": "Wastewater viral spike leads clinical cases by 1-2 weeks → diagnostic/pharma demand surge",
        "source": "CDC NWSS (hand-coded surge dates); yfinance",
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_vs_spy": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
        "events": event_results,
    })
    print(f"Done: {len(ok_events)} events, avg 30d return={avg_ret*100:.2f}%, win rate={win_rate*100:.0f}%")
    print(f"  Avg excess vs SPY: {avg_excess*100:.2f}%" if avg_excess is not None else "  No excess data")
    for e in ok_events:
        print(f"  {e['date']} {e['label']}: basket={e['basket_30d_return']*100:.2f}%, SPY={e.get('SPY_30d_return', 'N/A')}")


if __name__ == "__main__":
    main()
