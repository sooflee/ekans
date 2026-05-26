"""PL56_real_retail_sales_negative_long_xlp — Real Retail Sales YoY Negative → Long XLP
When RRSFS YoY turns negative after 12+ months positive: long XLP for 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL56_real_retail_sales_negative_long_xlp"
    try:
        fred = load_fred("RRSFS", start="1992-01-01")
        rrs = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if rrs.empty or len(rrs) < 24:
        return mark_failed(sid, "RRSFS data insufficient")

    # Compute YoY change
    yoy = rrs.pct_change(12) * 100
    yoy = yoy.dropna()

    # Find months where YoY first goes negative after 12+ positive months
    trigger_dates = []
    pos_count = 0
    for i in range(len(yoy)):
        if yoy.iloc[i] > 0:
            pos_count += 1
        else:
            if pos_count >= 12 and yoy.iloc[i] < 0:
                trigger_dates.append(yoy.index[i])
            pos_count = 0

    print(f"Real retail sales YoY negative (after 12+ positive months): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: YoY = {yoy.loc[d]:.2f}%")

    if len(trigger_dates) == 0:
        # Relax: just find any first negative after 6+ positive
        pos_count = 0
        for i in range(len(yoy)):
            if yoy.iloc[i] > 0:
                pos_count += 1
            else:
                if pos_count >= 6 and yoy.iloc[i] < 0:
                    trigger_dates.append(yoy.index[i])
                pos_count = 0
        print(f"Relaxed to 6+ positive: {len(trigger_dates)} events")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no real retail sales YoY negative events found")

    try:
        px = load_prices(["XLP", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "XLP" not in ret.columns or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

    xlp_ret = ret["XLP"]
    spy_ret = ret["SPY"]
    hold_days = 126

    pnl_series = pd.Series(0.0, index=xlp_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = xlp_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = xlp_ret.index[entry_mask][0]
        pos = xlp_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(xlp_ret))
        event_rets = xlp_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "yoy_pct": round(float(yoy.loc[td]), 2),
            "xlp_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Real Retail Sales Negative → Long XLP")
    rets_arr = [e["xlp_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long XLP for 126 days when real retail sales YoY turns negative after 12+ positive months",
        "mechanism": "Consumer spending contraction → staples outperform due to inelastic demand",
        "source": "FRED RRSFS; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
