"""
AI-3: Patent Cliff / Biosimilar Formulary Exclusion → Long MCOs (ELV + CI)

Hand-coded biosimilar formulary exclusion events → long ELV + CI equal-weight for 90 days.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AI-3"

# Hand-coded biosimilar formulary exclusion events
EVENTS = [
    "2023-01-03",  # Humira biosimilars launch; CVS/ESI exclude branded
    "2024-01-02",  # Additional Humira biosimilars enter, PBMs deepen exclusion
    "2025-01-02",  # Stelara (Yesintek) biosimilar enters at ~90% discount
    "2026-01-02",  # Full formulary exclusion of branded Stelara by all major PBMs
]

LONG_TICKERS = ["ELV", "CI"]
HOLD_DAYS = 90

def main():
    try:
        prices = load_prices(LONG_TICKERS + ["SPY"], start="2022-06-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data")
        print(f"FAILED {SIGNAL_ID}: no data")
        return

    missing = [t for t in LONG_TICKERS if t not in prices.columns]
    if missing:
        mark_failed(SIGNAL_ID, f"Missing tickers: {missing}")
        print(f"FAILED {SIGNAL_ID}: missing {missing}")
        return

    rets = daily_returns(prices)

    event_results = []
    all_pnl = pd.Series(dtype=float)

    for evt_date_str in EVENTS:
        evt_date = pd.Timestamp(evt_date_str)
        valid_dates = rets.index[rets.index >= evt_date]
        if len(valid_dates) < HOLD_DAYS:
            # Use whatever data we have if at least 30 days
            if len(valid_dates) >= 30:
                hold_rets = rets[LONG_TICKERS].loc[valid_dates].mean(axis=1).iloc[:HOLD_DAYS]
            else:
                event_results.append({"date": evt_date_str, "status": "skipped", "reason": "insufficient data"})
                continue
        else:
            entry_idx = rets.index.get_loc(valid_dates[0])
            hold_rets = rets[LONG_TICKERS].iloc[entry_idx:entry_idx + HOLD_DAYS].mean(axis=1)

        if len(hold_rets) < 30:
            event_results.append({"date": evt_date_str, "status": "skipped", "reason": "too few days"})
            continue

        cumulative_ret = (1 + hold_rets).prod() - 1

        event_results.append({
            "date": evt_date_str,
            "status": "ok",
            "return": float(cumulative_ret),
            "days_held": len(hold_rets),
        })

        all_pnl = pd.concat([all_pnl, hold_rets])

    if all_pnl.empty:
        mark_failed(SIGNAL_ID, "No valid events")
        print(f"FAILED {SIGNAL_ID}: no valid events")
        return

    bench_ret = rets["SPY"].reindex(all_pnl.index).dropna() if "SPY" in rets.columns else None
    metrics = compute_metrics(all_pnl, benchmark=bench_ret, name="Patent Cliff → Long ELV+CI")

    valid_events = [e for e in event_results if e.get("status") == "ok"]
    event_rets = [e["return"] for e in valid_events]
    avg_event_ret = np.mean(event_rets) if event_rets else 0
    median_event_ret = np.median(event_rets) if event_rets else 0
    win_rate_events = sum(1 for r in event_rets if r > 0) / len(event_rets) if event_rets else 0

    extra = {
        "status": "ok",
        "rule": "Biosimilar formulary exclusion event (Jan each year) → long ELV+CI equal-weight for 90 days",
        "mechanism": "Biosimilar entry + PBM formulary exclusion → MCO drug cost savings → margin expansion for managed care",
        "source": "PBM formulary exclusion lists, FDA biosimilar approvals, Express Scripts/CVS Caremark updates",
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
