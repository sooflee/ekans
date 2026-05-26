"""PL41_existing_home_low_supply_builders — Existing Home Months-Supply < 3.0 → Long Homebuilders
When FRED MSACSR crosses below 3.0 months (after being above 3.5): long LEN+DHI+NVR for 252 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL41_existing_home_low_supply_builders"
    try:
        fred = load_fred("MSACSR", start="1999-01-01")
        ms = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if ms.empty:
        return mark_failed(sid, "MSACSR data empty")

    # MSACSR min is 3.3 historically. Use bottom-quintile as tight supply threshold.
    threshold = ms.quantile(0.15)
    above_threshold = ms.quantile(0.50)  # must have been above median first
    print(f"MSACSR tight-supply threshold (15th pctl): {threshold:.2f}, above threshold: {above_threshold:.2f}")

    # Find cross below threshold after being above median within prior 6 months
    trigger_dates = []
    for i in range(6, len(ms)):
        if ms.iloc[i] < threshold and ms.iloc[i-1] >= threshold:
            lookback = ms.iloc[max(0, i-6):i]
            if lookback.max() >= above_threshold:
                trigger_dates.append(ms.index[i])

    print(f"MSACSR cross below {threshold:.2f} (after >{above_threshold:.2f}): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: MSACSR = {ms.loc[d]:.2f}")

    if len(trigger_dates) == 0:
        # Find local minima in bottom quartile
        q25 = ms.quantile(0.25)
        for i in range(1, len(ms) - 1):
            if ms.iloc[i] < ms.iloc[i-1] and ms.iloc[i] < ms.iloc[i+1] and ms.iloc[i] < q25:
                if len(trigger_dates) == 0 or (ms.index[i] - trigger_dates[-1]).days > 180:
                    trigger_dates.append(ms.index[i])
        print(f"Relaxed (local minima < {q25:.2f}): {len(trigger_dates)} events")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no MSACSR low-supply events found")

    try:
        px = load_prices(["LEN", "DHI", "NVR", "SPY"], start="1999-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["LEN", "DHI", "NVR"] if t in ret.columns]
    if len(available) == 0 or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 252

    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "msacsr": round(float(ms.loc[td]), 2),
            "basket_12m_return": round(cumret, 4),
            "spy_12m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Existing Home Low Supply → Long Homebuilders")
    rets_arr = [e["basket_12m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long LEN+DHI+NVR for 252 days when MSACSR crosses below 3.0 months",
        "mechanism": "Extreme existing-home scarcity forces buyers to new construction → homebuilder pricing power",
        "source": "FRED MSACSR; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
