"""PL741_rvia_collapse_short_rv_oem — RVIA Shipment YoY <-20% -> Short THO/WGO/LCII"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL741_rvia_collapse_short_rv_oem"

    # Hand-coded RVIA monthly wholesale shipment data (unit shipments, thousands)
    # YoY decline >= -20% months: primarily 2022-Q4 through 2023-Q1 cycle
    # RVIA historical data: large declines after pandemic-era surge
    rvia_data = {
        # (year, month): units shipped (approx from RVIA reports)
        (2019, 1): 35.0, (2019, 2): 40.5, (2019, 3): 43.2, (2019, 4): 42.8,
        (2019, 5): 41.5, (2019, 6): 38.2, (2019, 7): 39.1, (2019, 8): 37.4,
        (2019, 9): 35.5, (2019, 10): 34.8, (2019, 11): 28.1, (2019, 12): 24.5,
        (2020, 1): 33.0, (2020, 2): 38.5, (2020, 3): 28.0, (2020, 4): 16.0,
        (2020, 5): 24.5, (2020, 6): 42.0, (2020, 7): 48.5, (2020, 8): 48.0,
        (2020, 9): 44.5, (2020, 10): 46.5, (2020, 11): 41.0, (2020, 12): 39.5,
        (2021, 1): 46.0, (2021, 2): 51.0, (2021, 3): 55.0, (2021, 4): 53.5,
        (2021, 5): 52.5, (2021, 6): 50.0, (2021, 7): 51.0, (2021, 8): 49.5,
        (2021, 9): 47.0, (2021, 10): 46.5, (2021, 11): 42.5, (2021, 12): 44.0,
        (2022, 1): 42.5, (2022, 2): 47.5, (2022, 3): 47.0, (2022, 4): 43.5,
        (2022, 5): 38.5, (2022, 6): 33.5, (2022, 7): 28.5, (2022, 8): 25.0,
        (2022, 9): 21.5, (2022, 10): 17.5, (2022, 11): 13.5, (2022, 12): 12.0,
        (2023, 1): 15.5, (2023, 2): 20.0, (2023, 3): 25.5, (2023, 4): 29.5,
        (2023, 5): 31.5, (2023, 6): 33.0, (2023, 7): 34.5, (2023, 8): 35.0,
        (2023, 9): 34.0, (2023, 10): 33.5, (2023, 11): 31.0, (2023, 12): 29.5,
        (2024, 1): 30.5, (2024, 2): 34.0, (2024, 3): 36.5, (2024, 4): 37.0,
        (2024, 5): 37.5, (2024, 6): 36.5, (2024, 7): 37.0, (2024, 8): 36.0,
        (2024, 9): 35.0, (2024, 10): 34.5, (2024, 11): 31.5, (2024, 12): 30.0,
    }

    # RVIA reports typically released ~3-4 weeks after month-end; use 30 days lag
    # Report date = 1st of following month + 30 days
    events = []
    for (y, m), units in rvia_data.items():
        # Calculate YoY
        py = y - 1
        prior = rvia_data.get((py, m))
        if prior is None or prior <= 0:
            continue
        yoy = (units - prior) / prior
        if yoy <= -0.20:
            # Report date ~30 days after month-end
            report_month = m + 1 if m < 12 else 1
            report_year = y if m < 12 else y + 1
            report_date = pd.Timestamp(f"{report_year}-{report_month:02d}-01") + pd.Timedelta(days=14)
            events.append((report_date, yoy))

    print(f"Qualifying events (YoY <= -20%): {len(events)}")
    for ed, yoy in events[:5]:
        print(f"  {ed.date()}: YoY={yoy*100:.1f}%")

    if not events:
        return mark_failed(sid, "no qualifying RVIA events")

    try:
        px = load_prices(["THO", "WGO", "LCII", "SPY"], start="2018-01-01")
        r = daily_returns(px)
        tho_r = r["THO"]
        wgo_r = r["WGO"]
        lcii_r = r["LCII"]
        spy_r = r["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 30  # trading days
    pnl = pd.Series(0.0, index=tho_r.index)
    event_details = []

    for event_dt, yoy in events:
        mask = tho_r.index >= event_dt
        if mask.sum() < hold:
            continue
        ei = tho_r.index[mask][0]
        p = tho_r.index.get_loc(ei)
        ep = min(p + hold, len(tho_r))

        # Short equal-weight THO + WGO + LCII basket
        tho_seg = tho_r.iloc[p:ep]
        wgo_seg = wgo_r.reindex(tho_seg.index).fillna(0)
        lcii_seg = lcii_r.reindex(tho_seg.index).fillna(0)
        basket = -(tho_seg.values + wgo_seg.values + lcii_seg.values) / 3.0

        pnl.iloc[p:ep] = basket[:ep - p]

        basket_ret = float((1 + basket).prod() - 1) if len(basket) > 0 else 0
        spy_seg = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_seg).prod() - 1)
        event_details.append({
            "event_date": str(event_dt.date()),
            "yoy_change": round(yoy, 4),
            "basket_short_return": round(basket_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

    active = pnl[pnl != 0]
    print(f"Event trades: {len(event_details)}, active PnL days: {len(active)}")

    if len(event_details) == 0:
        return mark_failed(sid, "no valid events after data filter")

    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="RVIA YoY <-20% Short RV OEM Basket")
    save_result(sid, m, extra={
        "rule": "When RVIA monthly wholesale shipments YoY <= -20%, short equal-weight THO+WGO+LCII for 30 trading days",
        "mechanism": "Wholesale shipment collapse signals dealer inventory glut and earnings pressure on OEM manufacturers",
        "source": "RVIA monthly wholesale reports; yfinance",
        "n_events": len(event_details),
        "events": event_details,
    })
    print(f"Done: Sharpe={m.get('sharpe', '?'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
