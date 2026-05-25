"""
AF-1: GLP-1 Adoption Milestones → Short RMD (ResMed)
At each GLP-1 milestone: short RMD for 90 days.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AF-1"

# Key GLP-1 adoption milestones
EVENTS = [
    ("2023-08-08", "Novo Nordisk Wegovy STEP-1 obesity data widely published/discussed"),
    ("2023-11-08", "Lilly tirzepatide (Zepbound) FDA approval for obesity"),
    ("2024-06-21", "SURMOUNT-OSA results (63% AHI reduction with tirzepatide)"),
    ("2024-10-15", "Cumulative US GLP-1 patients cross ~6M milestone (estimated)"),
]

TICKER = "RMD"  # ResMed — thesis: GLP-1 reduces obesity → less sleep apnea → less CPAP demand
HOLD_DAYS = 90


def run():
    print(f"GLP-1 milestone events: {len(EVENTS)}")
    for d, desc in EVENTS:
        print(f"  {d}: {desc}")

    prices = load_prices([TICKER, "SPY"], start="2022-01-01")
    rets = daily_returns(prices)

    if TICKER not in rets.columns:
        mark_failed(SIGNAL_ID, f"Could not load {TICKER}", extra={
            "rule": "GLP-1 milestone → short RMD 90 days",
            "mechanism": "GLP-1 weight loss reduces sleep apnea prevalence → lower CPAP demand → RMD revenue headwind",
            "source": "FDA approvals / clinical trial results (hand-coded)",
        })
        return

    event_results = []
    all_pnl = []

    for event_date_str, desc in EVENTS:
        entry_date = pd.Timestamp(event_date_str)
        valid_dates = rets.index[rets.index >= entry_date]
        if len(valid_dates) < 10:
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        period_rets = rets.iloc[start_loc:end_loc + 1]

        # SHORT RMD
        rmd_daily = -period_rets[TICKER].fillna(0)
        spy_daily = period_rets["SPY"].fillna(0)

        total_ret = (1 + rmd_daily).prod() - 1
        spy_ret = (1 + spy_daily).prod() - 1
        rmd_raw = (1 + period_rets[TICKER].fillna(0)).prod() - 1

        event_results.append({
            "event_date": event_date_str,
            "description": desc,
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "short_rmd_return": float(total_ret),
            "rmd_long_return": float(rmd_raw),
            "spy_return": float(spy_ret),
            "excess_vs_spy": float(total_ret - spy_ret),
        })

        all_pnl.append(rmd_daily)

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No valid events", extra={
            "rule": "GLP-1 milestone → short RMD 90 days",
            "mechanism": "GLP-1 weight loss reduces sleep apnea prevalence → lower CPAP demand",
            "source": "FDA approvals / clinical data (hand-coded)",
        })
        return

    # Combine (average overlapping periods)
    combined = pd.concat(all_pnl, axis=1)
    combined_pnl = combined.mean(axis=1).sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AF-1 GLP-1 → Short RMD")
    print_metrics(metrics)

    print("\nPer-event results:")
    for er in event_results:
        print(f"  {er['event_date']}: short RMD {er['short_rmd_return']*100:.1f}% (RMD went {er['rmd_long_return']*100:+.1f}%), SPY {er['spy_return']*100:.1f}%")

    avg_ret = np.mean([e["short_rmd_return"] for e in event_results])
    win_rate = np.mean([1 if e["short_rmd_return"] > 0 else 0 for e in event_results])
    print(f"\n  Avg short-RMD return: {avg_ret*100:.1f}%, Win rate: {win_rate*100:.0f}%")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "GLP-1 adoption milestone → short RMD (ResMed) for 90 days",
        "mechanism": "GLP-1 agonists reduce obesity → less sleep apnea → secular headwind to CPAP device demand (RMD's core business)",
        "source": "FDA approvals, STEP/SURMOUNT trial publications, prescription data (hand-coded milestones)",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
