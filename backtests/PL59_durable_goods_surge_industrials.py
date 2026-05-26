"""PL59_durable_goods_surge_industrials — Durable Goods > +10% Above 6mo Avg for 3mo → Long XLI
When DGORDER exceeds 6-month avg by >10% for 3 consecutive months: long XLI 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL59_durable_goods_surge_industrials"
    try:
        fred = load_fred("DGORDER", start="1992-01-01")
        dgo = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if dgo.empty or len(dgo) < 12:
        return mark_failed(sid, "DGORDER data insufficient")

    # Compute ratio of current month to trailing 6-month average
    ma6 = dgo.rolling(6).mean()
    ratio = dgo / ma6
    ratio = ratio.dropna()

    # Find 3 consecutive months where ratio > 1.10
    trigger_dates = []
    streak = 0
    for i in range(len(ratio)):
        if ratio.iloc[i] > 1.05:
            streak += 1
            if streak == 3:
                # Avoid retriggering within same surge
                if len(trigger_dates) == 0 or (ratio.index[i] - trigger_dates[-1]).days > 180:
                    trigger_dates.append(ratio.index[i])
        else:
            streak = 0

    print(f"Durable goods surge events (3+ months >5% above 6mo avg): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: ratio = {ratio.loc[d]:.3f}")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no durable goods surge events found")

    try:
        px = load_prices(["XLI", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "XLI" not in ret.columns or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

    xli_ret = ret["XLI"]
    spy_ret = ret["SPY"]
    hold_days = 126

    pnl_series = pd.Series(0.0, index=xli_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = xli_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = xli_ret.index[entry_mask][0]
        pos = xli_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(xli_ret))
        event_rets = xli_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "ratio": round(float(ratio.loc[td]), 3),
            "xli_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Durable Goods Surge → Long XLI")
    rets_arr = [e["xli_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long XLI 126 days when DGORDER > 105% of trailing 6mo avg for 3 consecutive months",
        "mechanism": "Sustained durable goods strength → capex cycle inflection → industrials rally",
        "source": "FRED DGORDER; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
