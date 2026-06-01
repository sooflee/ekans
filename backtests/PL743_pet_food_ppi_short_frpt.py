"""PL743_pet_food_ppi_short_frpt — Pet Food PPI YoY Spike -> Short FRPT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL743_pet_food_ppi_short_frpt"

    try:
        # PCU311111311111 = PPI Dog & Cat Food industry series (long history)
        fred = load_fred("PCU311111311111", start="2013-01-01")
        ppi = fred.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if ppi.empty:
        return mark_failed(sid, "no FRED data")

    # Compute YoY
    ppi_monthly = ppi.resample("MS").last()
    yoy = ppi_monthly.pct_change(12)

    # Find qualifying months: YoY > 8%
    qualifying = yoy[yoy > 0.08].dropna()
    print(f"Qualifying months (PPI pet food YoY > 8%): {len(qualifying)}")

    # PPI released ~2 weeks after month-end; use release_date = month + 15 days
    events = []
    for date, val in qualifying.items():
        release_date = date + pd.DateOffset(days=45)  # ~mid next month
        events.append((release_date, float(val)))

    if not events:
        return mark_failed(sid, "no qualifying PPI events")

    try:
        px = load_prices(["FRPT", "SPY"], start="2016-01-01")
        r = daily_returns(px)
        frpt_r = r["FRPT"]
        spy_r = r["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 30  # trading days
    pnl = pd.Series(0.0, index=frpt_r.index)
    event_details = []

    for event_dt, yoy_val in events:
        mask = frpt_r.index >= event_dt
        if mask.sum() < hold:
            continue
        ei = frpt_r.index[mask][0]
        p = frpt_r.index.get_loc(ei)
        ep = min(p + hold, len(frpt_r))

        # Short FRPT
        seg = -frpt_r.iloc[p:ep]
        pnl.iloc[p:ep] = seg.values[:ep - p]

        frpt_ret = float((1 + frpt_r.iloc[p:ep]).prod() - 1)
        spy_seg = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_seg).prod() - 1)
        event_details.append({
            "event_date": str(event_dt.date()),
            "ppi_yoy": round(yoy_val, 4),
            "frpt_short_return": round(-frpt_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

    active = pnl[pnl != 0]
    print(f"Event trades: {len(event_details)}, active PnL days: {len(active)}")

    if len(event_details) == 0:
        return mark_failed(sid, "no valid events after data filter")

    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Pet Food PPI Spike Short FRPT")
    save_result(sid, m, extra={
        "rule": "When FRED WPU02830132 (PPI pet food) YoY > +8%, short FRPT for 30 trading days from release",
        "mechanism": "Input cost spike squeezes margins for premium pet food maker; FRPT most leveraged to raw material costs",
        "source": "FRED WPU02830132; yfinance",
        "n_events": len(event_details),
        "events": event_details,
    })
    print(f"Done: Sharpe={m.get('sharpe', '?'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
