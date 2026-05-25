"""
AH-1: Hydro Drought → Short Pacific NW Utility (AVA)

When Western US hydro is >15% below 10yr avg → short AVA (hydro-dependent Pacific NW utility)
from Jul 1 through Dec 31.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AH-1"

# Hand-coded drought years with trigger/exit dates
# AVA data may not go back to 2001
EVENTS = [
    ("2001-07-02", "2001-12-31"),  # Pacific NW power crisis
    ("2015-07-01", "2015-12-31"),  # California drought
    ("2021-07-01", "2021-12-31"),  # Western mega-drought
]

TICKER = "AVA"

def main():
    try:
        prices = load_prices(TICKER, start="2000-01-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data for AVA")
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

    for entry_str, exit_str in EVENTS:
        entry_date = pd.Timestamp(entry_str)
        exit_date = pd.Timestamp(exit_str)

        # Check if we have data for this period
        period_rets = rets[(rets.index >= entry_date) & (rets.index <= exit_date)]

        if len(period_rets) < 10:
            event_results.append({
                "entry": entry_str,
                "exit": exit_str,
                "status": "skipped",
                "reason": f"insufficient data (only {len(period_rets)} days)"
            })
            continue

        # Short position: PnL = negative of stock return
        short_pnl = -period_rets
        cumulative_ret = (1 + short_pnl).prod() - 1

        event_results.append({
            "entry": entry_str,
            "exit": exit_str,
            "status": "ok",
            "return": float(cumulative_ret),
            "days_held": len(period_rets),
        })

        all_pnl = pd.concat([all_pnl, short_pnl])

    if all_pnl.empty:
        mark_failed(SIGNAL_ID, "No valid events")
        print(f"FAILED {SIGNAL_ID}: no valid events")
        return

    metrics = compute_metrics(all_pnl, name="Hydro Drought → Short AVA")

    valid_events = [e for e in event_results if e.get("status") == "ok"]
    event_rets = [e["return"] for e in valid_events]
    avg_event_ret = np.mean(event_rets) if event_rets else 0
    median_event_ret = np.median(event_rets) if event_rets else 0
    win_rate_events = sum(1 for r in event_rets if r > 0) / len(event_rets) if event_rets else 0

    extra = {
        "status": "ok",
        "rule": "Western US hydro >15% below 10yr avg → short AVA (hydro-dependent utility) Jul-Dec",
        "mechanism": "Drought reduces hydro generation → higher purchased power costs + lower margins for hydro-dependent utilities",
        "source": "EIA monthly hydro generation, USBR reservoir levels, NOAA drought monitor",
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
