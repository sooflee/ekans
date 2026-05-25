"""
AE-2: Hyperscaler Capex Ratchet → Power-Infra Basket
When ≥3/4 hyperscalers report capex >20% YoY in a quarter: long {ETN, VRT, PWR, HUBB, AMSC, MOD} for 90 days.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AE-2"

# Hand-coded capex data and trigger dates (latest hyperscaler earnings report date per quarter)
# Each entry: (quarter, trigger_date, n_above_20pct, description)
TRIGGERS = [
    ("Q4-2023", "2024-02-01", 4, "MSFT +55%, GOOG +45%, META +35%, AMZN +20% → 4/4"),
    ("Q1-2024", "2024-04-25", 3, "MSFT +79%, GOOG +91%, AMZN +31% → 3/4 (META +5% miss)"),
    ("Q2-2024", "2024-07-30", 4, "MSFT +78%, GOOG +91%, META +31%, AMZN +54% → 4/4"),
    ("Q3-2024", "2024-10-30", 4, "MSFT +50%, GOOG +62%, META +36%, AMZN +81% → 4/4"),
    ("Q4-2024", "2025-01-30", 4, "All four >30% YoY → 4/4"),
]

BASKET = ["ETN", "VRT", "PWR", "HUBB", "AMSC", "MOD"]
HOLD_DAYS = 90


def run():
    print(f"Triggered events: {len(TRIGGERS)}")
    for q, d, n, desc in TRIGGERS:
        print(f"  {q} ({d}): {n}/4 above 20% — {desc}")

    # Load prices
    all_tickers = BASKET + ["SPY"]
    prices = load_prices(all_tickers, start="2023-01-01")
    rets = daily_returns(prices)

    event_results = []
    all_pnl = []

    for quarter, trigger_date, n_above, desc in TRIGGERS:
        entry_date = pd.Timestamp(trigger_date)
        valid_dates = rets.index[rets.index >= entry_date]
        if len(valid_dates) < 2:
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        period_rets = rets.iloc[start_loc:end_loc + 1]
        available = [t for t in BASKET if t in period_rets.columns and period_rets[t].notna().sum() > 0]
        if not available:
            continue

        basket_daily = period_rets[available].mean(axis=1)
        spy_daily = period_rets["SPY"] if "SPY" in period_rets.columns else pd.Series(0, index=period_rets.index)

        total_ret = (1 + basket_daily).prod() - 1
        spy_ret = (1 + spy_daily).prod() - 1

        event_results.append({
            "quarter": quarter,
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "tickers_used": available,
            "basket_return": float(total_ret),
            "spy_return": float(spy_ret),
            "excess": float(total_ret - spy_ret),
            "n_hyperscalers_above_20pct": n_above,
        })

        all_pnl.append(basket_daily)

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No events produced usable data", extra={
            "rule": "≥3/4 hyperscalers capex >20% YoY → long power-infra basket 90d",
            "mechanism": "AI capex surge drives demand for power/cooling infrastructure",
            "source": "10-Q filings (hand-coded)",
        })
        return

    # Combine all PnL (events may overlap — use mean for overlapping days)
    combined = pd.concat(all_pnl, axis=1)
    combined_pnl = combined.mean(axis=1)
    combined_pnl = combined_pnl.sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AE-2 Hyperscaler Capex → Power Infra")
    print_metrics(metrics)

    print("\nPer-event results:")
    for er in event_results:
        print(f"  {er['quarter']} ({er['entry']}→{er['exit']}): basket {er['basket_return']*100:.1f}%, SPY {er['spy_return']*100:.1f}%, excess {er['excess']*100:.1f}%")

    avg_ret = np.mean([e["basket_return"] for e in event_results])
    avg_excess = np.mean([e["excess"] for e in event_results])
    print(f"\n  Avg event return: {avg_ret*100:.1f}%, Avg excess: {avg_excess*100:.1f}%")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "≥3/4 hyperscalers report capex >20% YoY → long equal-weight {ETN,VRT,PWR,HUBB,AMSC,MOD} 90 days",
        "mechanism": "AI capex surge creates sustained demand for power infrastructure (transformers, switchgear, grid equipment)",
        "source": "Hyperscaler 10-Q filings (hand-coded capex figures)",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
