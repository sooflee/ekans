"""
AG-5: India Monsoon Failure → Long Sugar Futures
When IMD monsoon rainfall < 92% LPA: long SB=F (Sugar #11 futures) from Aug 1 through Nov 30.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AG-5"

# IMD monsoon seasons where deficit < 92% LPA
MONSOON_DEFICIT_YEARS = [
    (2014, 88, "Drought year — export ban Oct 2015"),
    (2015, 86, "Consecutive drought — ban continued"),
    # 2018 was 91% — borderline, include as test
    (2018, 91, "Borderline deficit (91% LPA)"),
]

TICKER = "SB=F"  # Sugar #11 futures
# Entry: Aug 1, Exit: Nov 30 of the deficit year


def run():
    print(f"Monsoon deficit years (< 92% LPA): {len(MONSOON_DEFICIT_YEARS)}")
    for year, pct, desc in MONSOON_DEFICIT_YEARS:
        print(f"  {year}: {pct}% LPA — {desc}")

    # Load sugar futures
    prices = load_prices([TICKER, "SPY"], start="2013-01-01")
    if TICKER not in prices.columns or prices[TICKER].dropna().empty:
        mark_failed(SIGNAL_ID, f"Could not load {TICKER} data from yfinance", extra={
            "rule": "IMD monsoon <92% LPA → long Sugar #11 Aug-Nov",
            "mechanism": "India monsoon failure reduces sugarcane yields → global sugar supply squeeze",
            "source": "IMD seasonal rainfall data (hand-coded)",
        })
        return

    rets = daily_returns(prices)

    event_results = []
    all_pnl = []

    for year, pct_lpa, desc in MONSOON_DEFICIT_YEARS:
        entry_date = pd.Timestamp(f"{year}-08-01")
        exit_date = pd.Timestamp(f"{year}-11-30")

        # Get period
        mask = (rets.index >= entry_date) & (rets.index <= exit_date)
        period_rets = rets.loc[mask]

        if len(period_rets) < 20:
            print(f"  {year}: insufficient data ({len(period_rets)} days)")
            continue

        if TICKER not in period_rets.columns or period_rets[TICKER].isna().all():
            print(f"  {year}: no sugar data")
            continue

        sugar_daily = period_rets[TICKER].fillna(0)
        spy_daily = period_rets["SPY"].fillna(0)

        total_ret = (1 + sugar_daily).prod() - 1
        spy_ret = (1 + spy_daily).prod() - 1

        event_results.append({
            "year": year,
            "monsoon_pct_lpa": pct_lpa,
            "description": desc,
            "entry": str(period_rets.index[0].date()),
            "exit": str(period_rets.index[-1].date()),
            "sugar_return": float(total_ret),
            "spy_return": float(spy_ret),
            "n_days": len(period_rets),
        })

        all_pnl.append(sugar_daily)
        print(f"  {year}: sugar {total_ret*100:.1f}% over {len(period_rets)} days")

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No valid monsoon-deficit events with sugar data", extra={
            "rule": "IMD monsoon <92% LPA → long Sugar #11 Aug-Nov",
            "mechanism": "India monsoon failure reduces sugarcane yields → global sugar supply squeeze",
            "source": "IMD seasonal rainfall data (hand-coded)",
        })
        return

    combined_pnl = pd.concat(all_pnl)
    combined_pnl = combined_pnl[~combined_pnl.index.duplicated(keep='first')]
    combined_pnl = combined_pnl.sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AG-5 Monsoon Failure → Sugar")
    print_metrics(metrics)

    print("\nPer-event summary:")
    for er in event_results:
        print(f"  {er['year']} ({er['monsoon_pct_lpa']}% LPA): sugar {er['sugar_return']*100:.1f}%")

    avg_ret = np.mean([e["sugar_return"] for e in event_results])
    print(f"\n  Avg seasonal return: {avg_ret*100:.1f}%")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "IMD monsoon rainfall <92% LPA → long Sugar #11 futures from Aug 1 through Nov 30",
        "mechanism": "India monsoon failure reduces sugarcane yields (India = #2 producer) → global sugar supply squeeze → price rise",
        "source": "IMD seasonal rainfall data (hand-coded deficit years)",
        "note": f"N={len(event_results)} events — low confidence",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
