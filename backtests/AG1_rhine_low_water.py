"""
AG-1: Rhine Low Water → Short BAS.DE, Long DOW (or LYB pre-2019)
When Kaub gauge drops below 100cm: short BASF, long US chemical proxy, hold 60 days.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AG-1"

# Rhine low-water events (Kaub gauge < 100cm sustained)
EVENTS = [
    ("2018-10-15", "Kaub dropped to 25cm (historic low)", "LYB"),   # DOW not yet trading (spun off Apr 2019)
    ("2022-08-12", "Kaub dropped to 32cm", "DOW"),
]

HOLD_DAYS = 60


def run():
    print(f"Rhine low-water events: {len(EVENTS)}")
    for d, desc, proxy in EVENTS:
        print(f"  {d}: {desc} (US proxy: {proxy})")

    # Load all needed tickers
    all_tickers = ["BAS.DE", "DOW", "LYB", "SPY"]
    prices = load_prices(all_tickers, start="2017-01-01")
    rets = daily_returns(prices)

    event_results = []
    all_pnl = []

    for event_date_str, desc, us_proxy in EVENTS:
        entry_date = pd.Timestamp(event_date_str)
        valid_dates = rets.index[rets.index >= entry_date]
        if len(valid_dates) < 10:
            print(f"  Skipping {event_date_str}: insufficient data")
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        period_rets = rets.iloc[start_loc:end_loc + 1]

        # Check data availability
        has_basf = "BAS.DE" in period_rets.columns and period_rets["BAS.DE"].notna().sum() > 10
        has_proxy = us_proxy in period_rets.columns and period_rets[us_proxy].notna().sum() > 10

        if not has_basf and not has_proxy:
            print(f"  Skipping {event_date_str}: no BASF or proxy data")
            continue

        # Long-short: short BAS.DE (50%), long US proxy (50%)
        daily_pnl = pd.Series(0.0, index=period_rets.index)
        leg_returns = {}

        if has_basf:
            basf_ret = period_rets["BAS.DE"].fillna(0)
            daily_pnl -= 0.5 * basf_ret  # SHORT
            leg_returns["SHORT_BAS.DE"] = float((1 + basf_ret).prod() - 1)

        if has_proxy:
            proxy_ret = period_rets[us_proxy].fillna(0)
            daily_pnl += 0.5 * proxy_ret  # LONG
            leg_returns[f"LONG_{us_proxy}"] = float((1 + proxy_ret).prod() - 1)

        total_ret = (1 + daily_pnl).prod() - 1
        spy_daily = period_rets["SPY"].fillna(0) if "SPY" in period_rets.columns else pd.Series(0, index=period_rets.index)
        spy_ret = (1 + spy_daily).prod() - 1

        event_results.append({
            "event_date": event_date_str,
            "description": desc,
            "us_proxy": us_proxy,
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "ls_return": float(total_ret),
            "spy_return": float(spy_ret),
            "excess_vs_spy": float(total_ret - spy_ret),
            "per_leg": leg_returns,
        })

        all_pnl.append(daily_pnl)
        print(f"  {event_date_str}: L/S {total_ret*100:.1f}%")

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No valid events (BAS.DE data may be unavailable)", extra={
            "rule": "Rhine Kaub gauge <100cm → short BAS.DE, long US chem (DOW/LYB) 60d",
            "mechanism": "Low Rhine water halts BASF Ludwigshafen barge logistics → production disruption; US chemicals unaffected",
            "source": "WSV Kaub gauge data (hand-coded events)",
        })
        return

    combined_pnl = pd.concat(all_pnl)
    combined_pnl = combined_pnl[~combined_pnl.index.duplicated(keep='first')]
    combined_pnl = combined_pnl.sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AG-1 Rhine Low Water Pair")
    print_metrics(metrics)

    print("\nPer-event results:")
    for er in event_results:
        print(f"  {er['event_date']}: L/S {er['ls_return']*100:.1f}%, SPY {er['spy_return']*100:.1f}%")
        for leg, r in er["per_leg"].items():
            print(f"    {leg}: {r*100:.1f}%")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "Rhine Kaub gauge <100cm sustained → short BAS.DE (50%) + long DOW/LYB (50%), hold 60 days",
        "mechanism": "Low Rhine water disrupts BASF's barge-dependent Ludwigshafen logistics; US chemical peers are unaffected → relative outperformance",
        "source": "WSV Kaub gauge records (hand-coded historic low-water events)",
        "note": f"N={len(event_results)} events — very low confidence",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
