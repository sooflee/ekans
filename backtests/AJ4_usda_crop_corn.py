"""
AJ-4: USDA Crop Progress G+E Cumulative Drop → Long Corn (ZC=F)

When USDA weekly corn "Good+Excellent" % drops >10pp from season's May baseline → long corn for 20 days.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AJ-4"

# Hand-coded trigger dates
EVENTS = [
    "2002-06-24",
    "2005-07-18",
    "2006-07-17",
    "2011-07-18",
    "2012-06-25",
    "2019-06-10",
    "2022-07-25",
    "2023-07-03",
]

HOLD_DAYS = 20
TICKER = "ZC=F"

def main():
    try:
        prices = load_prices(TICKER, start="2001-01-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data for ZC=F")
        print(f"FAILED {SIGNAL_ID}: no data")
        return

    if isinstance(prices, pd.DataFrame):
        px = prices.iloc[:, 0]
    else:
        px = prices

    px = px.dropna()
    rets = px.pct_change().dropna()

    event_results = []
    all_pnl = pd.Series(dtype=float)

    for evt_date_str in EVENTS:
        evt_date = pd.Timestamp(evt_date_str)
        valid_dates = rets.index[rets.index >= evt_date]
        if len(valid_dates) < HOLD_DAYS:
            event_results.append({"date": evt_date_str, "status": "skipped", "reason": "insufficient data"})
            continue

        entry_idx = rets.index.get_loc(valid_dates[0])
        hold_rets = rets.iloc[entry_idx:entry_idx + HOLD_DAYS]

        if len(hold_rets) < 5:
            event_results.append({"date": evt_date_str, "status": "skipped", "reason": "too few days"})
            continue

        cumulative_ret = (1 + hold_rets).prod() - 1
        entry_price = px.loc[valid_dates[0]]

        event_results.append({
            "date": evt_date_str,
            "status": "ok",
            "entry_price": float(entry_price),
            "return": float(cumulative_ret),
            "days_held": len(hold_rets),
        })

        all_pnl = pd.concat([all_pnl, hold_rets])

    if all_pnl.empty:
        mark_failed(SIGNAL_ID, "No valid events")
        print(f"FAILED {SIGNAL_ID}: no valid events")
        return

    metrics = compute_metrics(all_pnl, name="USDA G+E Drop → Long Corn")

    valid_events = [e for e in event_results if e.get("status") == "ok"]
    event_rets = [e["return"] for e in valid_events]
    avg_event_ret = np.mean(event_rets) if event_rets else 0
    median_event_ret = np.median(event_rets) if event_rets else 0
    win_rate_events = sum(1 for r in event_rets if r > 0) / len(event_rets) if event_rets else 0

    extra = {
        "status": "ok",
        "rule": "USDA weekly corn G+E % drops >10pp from May baseline → long ZC=F for 20 trading days",
        "mechanism": "Crop condition deterioration signals supply shortfall, corn prices rally on weather stress",
        "source": "USDA Crop Progress reports (weekly, May-Sep)",
        "n_events": len(valid_events),
        "avg_event_return": float(avg_event_ret),
        "median_event_return": float(median_event_ret),
        "event_win_rate": float(win_rate_events),
        "events": event_results,
    }

    save_result(SIGNAL_ID, metrics, extra)
    print_metrics(metrics)
    print(f"\nEvents: {len(valid_events)} valid / {len(EVENTS)} total")
    print(f"Avg event return: {avg_event_ret*100:.2f}%  Median: {median_event_ret*100:.2f}%  Win rate: {win_rate_events*100:.0f}%")

if __name__ == "__main__":
    main()
