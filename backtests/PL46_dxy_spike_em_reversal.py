"""PL46_dxy_spike_em_reversal — DXY 3-Month Spike > +8% → Long EEM After Rollover
When DTWEXBGS 63-day return > +8%, wait for rollover below +4%, then long EEM 63 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL46_dxy_spike_em_reversal"
    try:
        fred = load_fred("DTWEXBGS", start="2006-01-01")
        dxy = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if dxy.empty:
        return mark_failed(sid, "DTWEXBGS data empty")

    # Compute rolling 63-day return
    ret_63d = dxy / dxy.shift(63) - 1
    ret_63d = ret_63d.dropna()

    # Find spike events: 63d return > +8%
    # Then find rollover: 63d return falls below +4% after a spike
    trigger_dates = []
    in_spike = False
    for i in range(len(ret_63d)):
        if ret_63d.iloc[i] > 0.08:
            in_spike = True
        elif in_spike and ret_63d.iloc[i] < 0.04:
            trigger_dates.append(ret_63d.index[i])
            in_spike = False

    print(f"DXY spike+rollover events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: 63d return = {ret_63d.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no DXY spike+rollover events found")

    try:
        px = load_prices(["EEM", "SPY"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "EEM" not in ret.columns or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

    eem_ret = ret["EEM"]
    spy_ret = ret["SPY"]
    hold_days = 63

    pnl_series = pd.Series(0.0, index=eem_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_mask = eem_ret.index >= td
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = eem_ret.index[entry_mask][0]
        pos = eem_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(eem_ret))
        event_rets = eem_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "dxy_63d_ret": round(float(ret_63d.loc[td]), 4),
            "eem_3m_return": round(cumret, 4),
            "spy_3m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="DXY Spike → Long EEM")
    rets_arr = [e["eem_3m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long EEM 63 days after DXY 63d return spikes >+8% then rolls back below +4%",
        "mechanism": "EM equities mean-revert after forced-selling exhaustion from dollar strength",
        "source": "FRED DTWEXBGS; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
