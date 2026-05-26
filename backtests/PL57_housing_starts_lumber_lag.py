"""PL57_housing_starts_lumber_lag — Housing Starts Surge > +15% YoY → Long WOOD 3mo Later
When HOUST YoY > +15% for 2 consecutive months: wait 63 days, then long WOOD for 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL57_housing_starts_lumber_lag"
    try:
        fred = load_fred("HOUST", start="1990-01-01")
        houst = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if houst.empty or len(houst) < 24:
        return mark_failed(sid, "HOUST data insufficient")

    # Compute YoY change
    yoy = houst.pct_change(12)
    yoy = yoy.dropna()

    # Find 2 consecutive months with YoY > +15%
    trigger_dates = []
    for i in range(1, len(yoy)):
        if yoy.iloc[i] > 0.15 and yoy.iloc[i-1] > 0.15:
            # Avoid re-triggering within same surge
            if len(trigger_dates) == 0 or (yoy.index[i] - trigger_dates[-1]).days > 180:
                trigger_dates.append(yoy.index[i])

    print(f"Housing starts YoY > +15% for 2 months: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: YoY = {yoy.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no housing starts surge events found")

    try:
        px = load_prices(["WOOD", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "WOOD" not in ret.columns:
        return mark_failed(sid, "WOOD ETF data not available")

    wood_ret = ret["WOOD"]
    spy_ret = ret["SPY"]
    wait_days = 63
    hold_days = 126

    pnl_series = pd.Series(0.0, index=wood_ret.index)
    event_results = []

    for td in trigger_dates:
        # Wait 63 trading days after trigger
        entry_mask = wood_ret.index >= td
        if entry_mask.sum() < wait_days + hold_days:
            continue
        # Find actual entry point (63 trading days after trigger)
        first_available = wood_ret.index[entry_mask][0]
        first_pos = wood_ret.index.get_loc(first_available)
        entry_pos = first_pos + wait_days
        if entry_pos + hold_days > len(wood_ret):
            continue

        entry_idx = wood_ret.index[entry_pos]
        end_pos = min(entry_pos + hold_days, len(wood_ret))
        event_rets = wood_ret.iloc[entry_pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[entry_pos:end_pos] = event_rets.values[:end_pos - entry_pos]

        spy_event = spy_ret.iloc[entry_pos:end_pos]
        spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "houst_yoy": round(float(yoy.loc[td]), 4),
            "wood_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4),
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment (WOOD starts Jun 2008)")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Housing Starts → Long WOOD (lagged)")
    rets_arr = [e["wood_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long WOOD 126 days, entering 63 days after HOUST YoY > +15% for 2 consecutive months",
        "mechanism": "Housing starts surge → lumber demand peaks during framing phase 2-4 months later",
        "source": "FRED HOUST; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
