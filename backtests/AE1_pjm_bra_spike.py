"""
AE-1: PJM BRA Clearing Price Spike → Long Power Generators
When Base Residual Auction clears >100% above prior auction: long {CEG, VST, NRG, PSEG} 90 days.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AE-1"

# PJM BRA auction results (hand-coded from PJM archives)
# Format: (delivery_year, result_date, clearing_price_MW_day, prior_price)
BRA_AUCTIONS = [
    ("2023/24", "2020-07-20", 140.00, 76.53),   # +83% — borderline
    ("2024/25", "2021-06-15", 34.13, 140.00),    # -76% — no trigger
    ("2025/26", "2022-06-15", 28.92, 34.13),     # -15% — no trigger
    ("2026/27", "2024-06-18", 269.92, 28.92),    # +833% — TRIGGER
    ("2027/28", "2024-12-18", 285.00, 269.92),   # +6% — no trigger
]

THRESHOLD = 1.0  # >100% increase triggers

# Tickers: CEG only listed since Feb 2022 (Exelon spinoff)
# Use EXC as proxy pre-2022
TICKERS_POST_2022 = ["CEG", "VST", "NRG", "PSEG"]
TICKERS_PRE_2022 = ["EXC", "VST", "NRG", "PSEG"]

HOLD_DAYS = 90


def run():
    # Identify triggered events
    events = []
    for dy, date_str, price, prior in BRA_AUCTIONS:
        pct_change = (price - prior) / prior
        if pct_change > THRESHOLD:
            events.append({
                "delivery_year": dy,
                "date": pd.Timestamp(date_str),
                "price": price,
                "prior": prior,
                "pct_change": pct_change,
            })
        elif pct_change > 0.8:  # borderline
            events.append({
                "delivery_year": dy,
                "date": pd.Timestamp(date_str),
                "price": price,
                "prior": prior,
                "pct_change": pct_change,
                "borderline": True,
            })

    print(f"Triggered events: {len(events)}")
    for e in events:
        print(f"  {e['delivery_year']}: {e['date'].date()} ${e['price']:.2f} vs ${e['prior']:.2f} = +{e['pct_change']*100:.0f}%{'  (borderline)' if e.get('borderline') else ''}")

    # Load prices
    all_tickers = list(set(TICKERS_POST_2022 + TICKERS_PRE_2022 + ["SPY"]))
    prices = load_prices(all_tickers, start="2019-01-01")
    rets = daily_returns(prices)

    event_results = []
    all_pnl = []

    for e in events:
        entry_date = e["date"]
        # Find next trading day on or after entry date
        valid_dates = rets.index[rets.index >= entry_date]
        if len(valid_dates) == 0:
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        # Select tickers
        if entry_date.year >= 2022:
            tickers = TICKERS_POST_2022
        else:
            tickers = TICKERS_PRE_2022

        # Compute equal-weight basket return
        period_rets = rets.iloc[start_loc:end_loc + 1]
        available = [t for t in tickers if t in period_rets.columns and period_rets[t].notna().sum() > 0]
        if not available:
            continue

        basket_daily = period_rets[available].mean(axis=1)
        spy_daily = period_rets["SPY"] if "SPY" in period_rets.columns else pd.Series(0, index=period_rets.index)

        total_ret = (1 + basket_daily).prod() - 1
        spy_ret = (1 + spy_daily).prod() - 1

        event_results.append({
            "delivery_year": e["delivery_year"],
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "tickers": available,
            "basket_return": float(total_ret),
            "spy_return": float(spy_ret),
            "excess": float(total_ret - spy_ret),
            "borderline": e.get("borderline", False),
            "pct_change_trigger": float(e["pct_change"]),
        })

        all_pnl.append(basket_daily)

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No events triggered", extra={
            "rule": "BRA price >100% vs prior → long power gen basket 90d",
            "mechanism": "Capacity price spike signals power scarcity, benefits generators",
            "source": "PJM BRA archives",
        })
        return

    # Combine into single PnL series (concatenate event windows)
    combined_pnl = pd.concat(all_pnl)
    # Remove any overlapping dates (keep first occurrence)
    combined_pnl = combined_pnl[~combined_pnl.index.duplicated(keep='first')]
    combined_pnl = combined_pnl.sort_index()

    # Also load SPY for benchmark
    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AE-1 PJM BRA Spike")
    print_metrics(metrics)

    for er in event_results:
        print(f"\n  Event {er['delivery_year']}: basket {er['basket_return']*100:.1f}%, SPY {er['spy_return']*100:.1f}%, excess {er['excess']*100:.1f}%{'  [borderline]' if er['borderline'] else ''}")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "BRA clearing price >100% vs prior auction → long equal-weight {CEG,VST,NRG,PSEG} for 90 days",
        "mechanism": "Capacity price spike signals structural power scarcity, directly benefits generator revenues",
        "source": "PJM BRA archives (hand-coded)",
        "note": "Only 1-2 events; very low confidence due to small N",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
