"""
AQ-1: Copper Near Marginal Cost Floor → Long HG=F
Hand-coded periods when copper spot was within 15% of estimated 90th-percentile AISC.
Long HG=F (copper futures) from trigger date, hold 12 months.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AQ-1"
HOLD_MONTHS = 12
COPPER_TICKER = "HG=F"

# (trigger_date, description, copper_price_at_trigger, est_floor)
TRIGGER_EVENTS = [
    ("2016-01-15", "Copper $1.94/lb — below est floor ~$2.50", 1.94, 2.50),
    ("2020-03-23", "Copper $2.10/lb — COVID crash, well below floor", 2.10, 2.80),
    ("2022-07-15", "Copper $3.30/lb — near ~$3.80 floor (~13% within)", 3.30, 3.80),
]


def run():
    print("=== AQ-1: Copper Near Marginal Cost → Long Copper Futures ===\n")
    for d, desc, price, floor in TRIGGER_EVENTS:
        pct = (price - floor) / floor * 100
        print(f"  {d}: {desc} ({pct:+.1f}% vs floor)")

    # --- Load copper futures ---
    tickers = [COPPER_TICKER, "SPY"]
    prices = load_prices(tickers, start="2010-01-01")
    rets = daily_returns(prices)
    print(f"\nPrice data: {prices.index[0].date()} → {prices.index[-1].date()}")

    if COPPER_TICKER not in prices.columns or prices[COPPER_TICKER].dropna().empty:
        # Try alternative ticker
        print("HG=F not available, trying HG=F with different date range...")
        mark_failed(SIGNAL_ID, "Copper futures data unavailable from yfinance",
                     extra={"rule": "Copper near 90th pct AISC → long copper futures 12mo",
                            "mechanism": "Supply-side floor: when copper nears marginal cost, high-cost mines shut → supply contracts → price recovers",
                            "source": "Hand-coded triggers + yfinance HG=F"})
        return

    # --- Build trade windows ---
    trade_pnls = []
    trade_details = []
    for trigger_str, desc, price_at, floor_est in TRIGGER_EVENTS:
        entry = pd.Timestamp(trigger_str)
        exit_date = entry + pd.DateOffset(months=HOLD_MONTHS)

        mask = (rets.index >= entry) & (rets.index <= exit_date)
        window_rets = rets.loc[mask]

        if COPPER_TICKER not in window_rets.columns or window_rets[COPPER_TICKER].notna().sum() < 20:
            print(f"\n  {entry.date()}: copper data insufficient, skipping")
            continue

        port_ret = window_rets[COPPER_TICKER].fillna(0)
        cum = (1 + port_ret).cumprod()
        total_ret = cum.iloc[-1] - 1

        spy_ret_window = window_rets["SPY"].fillna(0) if "SPY" in window_rets.columns else None
        spy_total = (1 + spy_ret_window).cumprod().iloc[-1] - 1 if spy_ret_window is not None else 0

        # Get actual copper price at exit
        exit_mask = prices.index <= exit_date
        if exit_mask.any():
            exit_price = prices.loc[exit_mask, COPPER_TICKER].dropna().iloc[-1]
        else:
            exit_price = np.nan

        trade_details.append({
            "trigger": trigger_str,
            "description": desc,
            "copper_at_entry": price_at,
            "est_floor": floor_est,
            "copper_at_exit": float(exit_price) if not np.isnan(exit_price) else None,
            "total_return": float(total_ret),
            "spy_return": float(spy_total),
            "excess": float(total_ret - spy_total),
            "n_days": len(port_ret),
        })
        trade_pnls.append(port_ret)
        print(f"\n  {entry.date()} (Cu @ ${price_at:.2f}):")
        print(f"    12mo return: {total_ret*100:+.1f}% | SPY: {spy_total*100:+.1f}% | Excess: {(total_ret-spy_total)*100:+.1f}%")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows with copper data",
                     extra={"rule": "Copper near 90th pct AISC → long copper 12mo",
                            "mechanism": "Supply-side floor: marginal cost support",
                            "source": "Hand-coded triggers + yfinance HG=F"})
        return

    # --- Combine ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="AQ-1 Copper Marginal Cost Floor")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Copper within 15% of est 90th-pct AISC → long HG=F, hold 12 months",
        "mechanism": "When copper nears marginal cost of production, high-cost mines curtail → supply contraction → price recovery. Fundamental floor.",
        "source": "Hand-coded trigger dates + yfinance HG=F",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
