"""
AJ-2: Peru Anchovy Quota Shortfall → Long Soybean Meal (ZM=F)

When Peru PRODUCE sets quota >30% below prior year → long ZM=F for 60 trading days.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AJ-2"

# Hand-coded quota shortfall events
EVENTS = [
    "2014-04-01",  # first season quota -40% vs 2013
    "2014-11-20",  # second season cancelled entirely
    "2017-06-28",  # first season quota suspended due to El Niño
    "2020-11-13",  # second season significantly reduced
    "2023-06-08",  # first season cancelled — fishmeal output -23%
    "2026-04-15",  # first season quota -36% YoY at 1.91M MT
]

HOLD_DAYS = 60
TICKER = "ZM=F"

def main():
    try:
        prices = load_prices(TICKER, start="2013-01-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data for ZM=F")
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
        # Find next available trading day on or after event date
        valid_dates = rets.index[rets.index >= evt_date]
        if len(valid_dates) < HOLD_DAYS:
            event_results.append({"date": evt_date_str, "status": "skipped", "reason": "insufficient data after event"})
            continue

        entry_idx = rets.index.get_loc(valid_dates[0])
        hold_rets = rets.iloc[entry_idx:entry_idx + HOLD_DAYS]

        if len(hold_rets) < 10:
            event_results.append({"date": evt_date_str, "status": "skipped", "reason": "too few days"})
            continue

        cumulative_ret = (1 + hold_rets).prod() - 1
        entry_price = px.loc[valid_dates[0]]
        exit_price = px.iloc[entry_idx + len(hold_rets) - 1] if entry_idx + HOLD_DAYS <= len(px) else px.iloc[-1]

        event_results.append({
            "date": evt_date_str,
            "status": "ok",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "return": float(cumulative_ret),
            "days_held": len(hold_rets),
        })

        all_pnl = pd.concat([all_pnl, hold_rets])

    if all_pnl.empty:
        mark_failed(SIGNAL_ID, "No valid events produced PnL")
        print(f"FAILED {SIGNAL_ID}: no valid events")
        return

    # Compute metrics on concatenated PnL
    metrics = compute_metrics(all_pnl, name="Peru Anchovy → Long ZM=F")

    # Also compute simple avg/median per-event return
    valid_events = [e for e in event_results if e.get("status") == "ok"]
    event_rets = [e["return"] for e in valid_events]
    avg_event_ret = np.mean(event_rets) if event_rets else 0
    median_event_ret = np.median(event_rets) if event_rets else 0
    win_rate_events = np.mean([1 for r in event_rets if r > 0]) / len(event_rets) if event_rets else 0

    extra = {
        "status": "ok",
        "rule": "Peru PRODUCE sets anchovy quota >30% below prior year → long ZM=F for 60 trading days",
        "mechanism": "Reduced fishmeal supply from Peru raises soybean meal as substitute protein in feed markets",
        "source": "Peru PRODUCE quota announcements, historical fishmeal production data",
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
