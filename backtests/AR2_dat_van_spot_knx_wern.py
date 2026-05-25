"""
AR-2: DAT Van Spot Breakeven Cross → Long KNX/WERN
Proxy: same freight cycle trough timing as AR-1.
Hand-coded trough dates → long KNX+WERN equal-weight, hold 12 months.
Note: KNX was Knight Transportation pre-2017 merger — ticker may have limited history.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AR-2"
LONG_BASKET = ["KNX", "WERN"]
HOLD_MONTHS = 12

# Known freight recession trough dates (same as AR-1)
TROUGH_DATES = [
    ("2009-03-01", "GFC trough"),
    ("2016-03-01", "Industrial recession trough"),
    ("2019-09-01", "Freight recession trough"),
    ("2024-10-01", "Current cycle trough estimate"),
]


def run():
    print("=== AR-2: DAT Van Spot Breakeven Cross → Long KNX/WERN ===\n")
    for d, desc in TROUGH_DATES:
        print(f"  {d}: {desc}")

    # --- Load equity prices ---
    all_tickers = LONG_BASKET + ["SPY"]
    prices = load_prices(all_tickers, start="2005-01-01")
    rets = daily_returns(prices)
    print(f"\nPrice data: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Check ticker availability
    for t in LONG_BASKET:
        if t in prices.columns:
            first = prices[t].dropna().index
            if len(first) > 0:
                print(f"  {t}: from {first[0].date()} to {first[-1].date()}")
            else:
                print(f"  {t}: NO DATA")
        else:
            print(f"  {t}: not in download")

    # --- Build trade windows ---
    trade_pnls = []
    trade_details = []
    for trough_str, desc in TROUGH_DATES:
        entry = pd.Timestamp(trough_str)
        exit_date = entry + pd.DateOffset(months=HOLD_MONTHS)

        mask = (rets.index >= entry) & (rets.index <= exit_date)
        window_rets = rets.loc[mask]

        avail = [t for t in LONG_BASKET if t in window_rets.columns and window_rets[t].notna().sum() > 20]
        if not avail:
            print(f"\n  {entry.date()}: no ticker data available, skipping")
            # Try WERN-only if KNX unavailable (pre-merger)
            if "WERN" in window_rets.columns and window_rets["WERN"].notna().sum() > 20:
                avail = ["WERN"]
                print(f"    Falling back to WERN-only")
            else:
                continue

        port_ret = window_rets[avail].mean(axis=1)
        bench_ret = window_rets["SPY"] if "SPY" in window_rets.columns else None

        cum = (1 + port_ret).cumprod()
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        spy_cum = (1 + bench_ret).cumprod() if bench_ret is not None else None
        spy_total = spy_cum.iloc[-1] - 1 if spy_cum is not None and len(spy_cum) > 0 else 0

        trade_details.append({
            "trigger": trough_str,
            "description": desc,
            "entry": str(entry.date()),
            "exit": str(exit_date.date()),
            "tickers": avail,
            "total_return": float(total_ret),
            "spy_return": float(spy_total),
            "excess": float(total_ret - spy_total),
            "n_days": len(port_ret),
        })
        trade_pnls.append(port_ret)
        print(f"\n  {entry.date()} ({desc}): {avail} → {total_ret*100:+.1f}% (SPY {spy_total*100:+.1f}%, excess {(total_ret-spy_total)*100:+.1f}%)")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows",
                     extra={"rule": "Freight trough → long KNX+WERN 12mo",
                            "mechanism": "Spot rates cross breakeven from below → truckload carriers see earnings recovery",
                            "source": "Hand-coded trough dates + yfinance"})
        return

    # --- Combine ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="AR-2 DAT Van Spot → Long TL Carriers")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Freight cycle trough (DAT van spot proxy) → long KNX+WERN equal-weight 12mo",
        "mechanism": "When spot rates cross carrier breakeven from below, truckload earnings inflect → equities re-rate",
        "source": "Hand-coded trough dates (proxy for DAT van spot) + yfinance",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
