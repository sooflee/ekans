"""
AG-2: Transformer Shortage → OEM vs Utility Pair Trade
Regime trade: long {ETN, HUBB} vs short {NEE, AEP} from Jul 2022 (DOE >100-week lead times).
Also test from Jan 2023.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AG-2"

LONG_TICKERS = ["ETN", "HUBB"]
SHORT_TICKERS = ["NEE", "AEP"]

# Two regime start dates to test
REGIME_STARTS = [
    ("2022-07-01", "DOE first reports >100-week transformer lead times"),
    ("2023-01-03", "Transformer shortage widely discussed in utility earnings calls"),
]


def run():
    all_tickers = LONG_TICKERS + SHORT_TICKERS + ["SPY"]
    prices = load_prices(all_tickers, start="2021-01-01")
    rets = daily_returns(prices)

    results_by_start = []

    for start_date_str, desc in REGIME_STARTS:
        start_date = pd.Timestamp(start_date_str)
        valid_dates = rets.index[rets.index >= start_date]
        if len(valid_dates) < 30:
            continue

        period_rets = rets.loc[valid_dates]

        # Equal-weight long-short: long 50% each ETN/HUBB, short 50% each NEE/AEP
        daily_pnl = pd.Series(0.0, index=period_rets.index)
        for t in LONG_TICKERS:
            if t in period_rets.columns:
                daily_pnl += 0.5 * period_rets[t].fillna(0)
        for t in SHORT_TICKERS:
            if t in period_rets.columns:
                daily_pnl -= 0.5 * period_rets[t].fillna(0)

        spy_rets = rets["SPY"].reindex(period_rets.index)
        metrics = compute_metrics(daily_pnl, benchmark=spy_rets, name=f"AG-2 from {start_date_str}")
        print_metrics(metrics)

        # Per-leg cumulative returns
        leg_cum = {}
        for t in LONG_TICKERS + SHORT_TICKERS:
            if t in period_rets.columns:
                cum = (1 + period_rets[t].fillna(0)).prod() - 1
                leg_cum[t] = float(cum)

        results_by_start.append({
            "start_date": start_date_str,
            "description": desc,
            "metrics": metrics,
            "per_leg_cumulative": leg_cum,
        })

        print(f"\n  Per-leg cumulative returns from {start_date_str}:")
        for t, r in leg_cum.items():
            side = "LONG" if t in LONG_TICKERS else "SHORT"
            print(f"    {side} {t}: {r*100:.1f}%")

    if not results_by_start:
        mark_failed(SIGNAL_ID, "Insufficient data", extra={
            "rule": "Transformer lead time >100 weeks → long OEMs vs short utilities",
            "mechanism": "Shortage benefits equipment manufacturers at expense of utilities facing capex delays",
            "source": "DOE transformer report + utility earnings calls",
        })
        return

    # Use the Jul 2022 start as primary
    primary = results_by_start[0]
    primary_metrics = primary["metrics"]

    extra = {
        "status": "ok",
        "regime_trade": True,
        "results_by_start_date": results_by_start,
        "rule": "Transformer lead time >100 weeks (sustained from mid-2022) → long {ETN,HUBB} vs short {NEE,AEP}",
        "mechanism": "Equipment manufacturers benefit from pricing power during shortage; utilities face delayed projects and higher capex",
        "source": "DOE transformer availability report (Jul 2022), utility earnings calls",
    }

    save_result(SIGNAL_ID, primary_metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
