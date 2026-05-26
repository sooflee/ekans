"""PL5_housing_permits_completions_aggregate — Housing Permits/Completions Ratio → Long Aggregates
When PERMIT/COMPUTSA ratio crosses above 1.3, long equal-weight VMC+MLM+EXP for 6 months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL5_housing_permits_completions_aggregate"

    # Load FRED data for PERMIT and COMPUTSA
    try:
        from harness import load_fred
        fred_df = load_fred(["PERMIT", "COMPUTSA"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if fred_df is None or fred_df.empty:
        return mark_failed(sid, "FRED PERMIT/COMPUTSA data empty")

    # Compute ratio
    ratio = fred_df["PERMIT"] / fred_df["COMPUTSA"]
    ratio = ratio.dropna()

    if len(ratio) < 12:
        return mark_failed(sid, "insufficient FRED ratio data")

    # Find months where ratio crosses above 1.3
    cross_above = (ratio > 1.3) & (ratio.shift(1) <= 1.3)
    trigger_dates = ratio.index[cross_above]

    print(f"PERMIT/COMPUTSA ratio > 1.3 cross-above events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: ratio = {ratio.loc[d]:.3f}")

    if len(trigger_dates) == 0:
        # Also check if ratio is persistently above 1.3
        above_mask = ratio > 1.3
        if above_mask.any():
            # Use first month of each continuous period above 1.3
            transitions = above_mask.astype(int).diff()
            trigger_dates = ratio.index[transitions == 1]
            if len(trigger_dates) == 0:
                # Ratio might start above 1.3
                if above_mask.iloc[0]:
                    trigger_dates = pd.DatetimeIndex([ratio.index[0]])

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no PERMIT/COMPUTSA ratio cross-above-1.3 events found")

    # Load equity prices
    try:
        px = load_prices(["VMC", "MLM", "EXP", "SPY"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if any(t not in ret.columns for t in ["VMC", "MLM", "EXP", "SPY"]):
        return mark_failed(sid, f"missing tickers: {[t for t in ['VMC','MLM','EXP','SPY'] if t not in ret.columns]}")

    # Aggregate basket: equal-weight VMC+MLM+EXP
    basket_ret = (ret["VMC"] + ret["MLM"] + ret["EXP"]) / 3
    spy_ret = ret["SPY"]

    hold_days = 126  # ~6 months
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for td in trigger_dates:
        # Entry: first trading day of the month after the trigger
        entry_mask = basket_ret.index >= td
        if entry_mask.sum() < hold_days + 1:
            event_results.append({"trigger_date": str(td.date()), "ratio": round(float(ratio.loc[td]), 3), "status": "skipped"})
            continue

        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))

        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = np.maximum(pnl_series.iloc[pos:end_pos].values, 0) + event_rets.values[:end_pos - pos]
        # More accurate: just mark as in-position
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        # SPY same period
        spy_mask2 = spy_ret.index >= entry_idx
        if spy_mask2.sum() >= hold_days:
            spy_start = spy_ret.index.get_loc(spy_ret.index[spy_mask2][0])
            spy_event = spy_ret.iloc[spy_start:spy_start + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        event_results.append({
            "trigger_date": str(td.date()),
            "ratio": round(float(ratio.loc[td]), 3),
            "status": "ok",
            "basket_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e.get("status") == "ok"]
    if len(ok_events) < 1:
        return mark_failed(sid, "no valid events after equity data alignment")

    basket_rets_arr = np.array([e["basket_6m_return"] for e in ok_events])
    avg_ret = float(basket_rets_arr.mean())
    win_rate = float((basket_rets_arr > 0).mean())

    # Compute metrics on full PnL series
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Permits/Completions → Long Aggregates")
    else:
        m = {
            "name": "Permits/Completions → Long Aggregates",
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
        "rule": "Long equal-weight VMC+MLM+EXP for 126 trading days when PERMIT/COMPUTSA ratio crosses above 1.3",
        "mechanism": "Rising permits-to-completions ratio → construction backlog → guaranteed aggregate/cement demand",
        "source": "FRED PERMIT, COMPUTSA; yfinance",
        "n_trigger_events": len(ok_events),
        "avg_event_6m_return": round(avg_ret, 4),
        "event_win_rate": round(win_rate, 4),
        "events": event_results,
    })
    print(f"Done: {len(ok_events)} events, avg 6m return={avg_ret*100:.2f}%, win rate={win_rate*100:.0f}%")
    for e in ok_events:
        print(f"  {e['trigger_date']} (ratio={e['ratio']}): basket={e['basket_6m_return']*100:.2f}%, SPY={e.get('spy_6m_return', 'N/A')}")


if __name__ == "__main__":
    main()
