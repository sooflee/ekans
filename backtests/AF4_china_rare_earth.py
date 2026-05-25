"""
AF-4: China Rare Earth Export Controls → MP/TSLA vs F/GM
Long (MP 50% + TSLA 25%) vs Short (F 25% + GM 25%), hold 60 days after each restriction event.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AF-4"

# MOFCOM rare-earth / critical mineral restriction events
EVENTS = [
    ("2023-07-03", "China restricts Gallium + Germanium exports"),
    ("2023-10-20", "China restricts graphite exports"),
    ("2024-12-03", "China bans Gallium/Germanium/Antimony exports to US"),
    ("2025-04-04", "China restricts rare earth processing tech + 7 heavy RE elements"),
]

# Long-short weights
LONG_WEIGHTS = {"MP": 0.50, "TSLA": 0.25}
SHORT_WEIGHTS = {"F": 0.25, "GM": 0.25}  # short → negate
HOLD_DAYS = 60


def run():
    print(f"Rare earth restriction events: {len(EVENTS)}")
    for d, desc in EVENTS:
        print(f"  {d}: {desc}")

    all_tickers = list(set(list(LONG_WEIGHTS.keys()) + list(SHORT_WEIGHTS.keys()) + ["SPY"]))
    prices = load_prices(all_tickers, start="2022-01-01")
    rets = daily_returns(prices)

    event_results = []
    all_pnl = []

    for event_date_str, desc in EVENTS:
        entry_date = pd.Timestamp(event_date_str)
        valid_dates = rets.index[rets.index >= entry_date]
        if len(valid_dates) < 5:
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        period_rets = rets.iloc[start_loc:end_loc + 1]

        # Compute long-short daily PnL
        daily_pnl = pd.Series(0.0, index=period_rets.index)

        leg_returns = {}

        # Long legs
        for ticker, weight in LONG_WEIGHTS.items():
            if ticker in period_rets.columns:
                daily_pnl += weight * period_rets[ticker].fillna(0)
                leg_returns[f"LONG_{ticker}"] = float((1 + period_rets[ticker].fillna(0)).prod() - 1)

        # Short legs (negate)
        for ticker, weight in SHORT_WEIGHTS.items():
            if ticker in period_rets.columns:
                daily_pnl -= weight * period_rets[ticker].fillna(0)
                leg_returns[f"SHORT_{ticker}"] = float((1 + period_rets[ticker].fillna(0)).prod() - 1)

        total_ret = (1 + daily_pnl).prod() - 1
        spy_daily = period_rets["SPY"].fillna(0)
        spy_ret = (1 + spy_daily).prod() - 1

        event_results.append({
            "event_date": event_date_str,
            "description": desc,
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "ls_return": float(total_ret),
            "spy_return": float(spy_ret),
            "excess_vs_spy": float(total_ret - spy_ret),
            "per_leg": leg_returns,
        })

        all_pnl.append(daily_pnl)

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No usable events", extra={
            "rule": "China RE export control → long MP/TSLA vs short F/GM 60d",
            "mechanism": "RE restrictions benefit domestic producers (MP) and vertically integrated EV (TSLA) vs import-dependent legacy auto",
            "source": "MOFCOM announcements (hand-coded)",
        })
        return

    # Combine (average overlapping days)
    combined = pd.concat(all_pnl, axis=1)
    combined_pnl = combined.mean(axis=1).sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AF-4 China RE → MP/TSLA vs F/GM")
    print_metrics(metrics)

    print("\nPer-event results:")
    for er in event_results:
        print(f"  {er['event_date']}: L/S {er['ls_return']*100:.1f}%, SPY {er['spy_return']*100:.1f}%, excess {er['excess_vs_spy']*100:.1f}%")
        for leg, r in er["per_leg"].items():
            print(f"    {leg}: {r*100:.1f}%")

    avg_ls = np.mean([e["ls_return"] for e in event_results])
    print(f"\n  Avg L/S return per event: {avg_ls*100:.1f}%")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "China rare earth/critical mineral export restriction → long (MP 50%, TSLA 25%) vs short (F 25%, GM 25%) for 60 days",
        "mechanism": "RE export controls benefit domestic RE producers and vertically-integrated EV makers while hurting import-dependent legacy automakers",
        "source": "MOFCOM announcements (hand-coded dates)",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
