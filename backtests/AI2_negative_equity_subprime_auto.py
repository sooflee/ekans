"""
AI-2: Negative Equity + Manheim Decline → Short Subprime Auto Lenders (SC + CACC)

Regime trade: short SC + CACC equal-weight from Jan 2024 through present.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AI-2"

SHORT_TICKERS = ["CACC", "ALLY"]  # SC (Santander Consumer) was taken private early 2024; use ALLY as substitute
BENCHMARK = "SPY"
START_DATE = "2024-01-02"

def main():
    all_tickers = SHORT_TICKERS + [BENCHMARK]
    try:
        prices = load_prices(all_tickers, start="2023-06-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data")
        print(f"FAILED {SIGNAL_ID}: no data")
        return

    missing = [t for t in SHORT_TICKERS if t not in prices.columns]
    if missing:
        mark_failed(SIGNAL_ID, f"Missing tickers: {missing}")
        print(f"FAILED {SIGNAL_ID}: missing {missing}")
        return

    rets = daily_returns(prices)
    rets = rets[rets.index >= START_DATE]

    # Short position PnL = negative of stock return
    short_basket_ret = rets[SHORT_TICKERS].mean(axis=1)
    short_pnl = -short_basket_ret
    short_pnl = short_pnl.dropna()

    if len(short_pnl) < 30:
        mark_failed(SIGNAL_ID, "Insufficient data")
        print(f"FAILED {SIGNAL_ID}: insufficient data")
        return

    bench_ret = rets[BENCHMARK].reindex(short_pnl.index).dropna() if BENCHMARK in rets.columns else None
    metrics = compute_metrics(short_pnl, benchmark=bench_ret, name="Negative Equity → Short SC+CACC")

    cum_ret = (1 + short_pnl).cumprod().iloc[-1] - 1
    ally_cum = -((1 + rets["ALLY"]).cumprod().iloc[-1] - 1) if "ALLY" in rets.columns else None
    cacc_cum = -((1 + rets["CACC"]).cumprod().iloc[-1] - 1) if "CACC" in rets.columns else None

    extra = {
        "status": "ok",
        "rule": "Short CACC + ALLY equal-weight from Jan 2024 (Fitch subprime auto ABS 60+ DQ > 6%). SC taken private.",
        "mechanism": "Manheim index declining + negative equity at records → subprime auto credit deterioration → lender losses",
        "source": "Fitch subprime auto ABS tracker, Manheim Used Vehicle Value Index, NY Fed auto loan data",
        "trade_type": "regime_short",
        "start_date": START_DATE,
        "short_basket_cum_return": float(cum_ret),
        "ALLY_short_return": float(ally_cum) if ally_cum is not None else None,
        "CACC_short_return": float(cacc_cum) if cacc_cum is not None else None,
    }

    save_result(SIGNAL_ID, metrics, extra)
    print_metrics(metrics)
    print(f"\nShort basket cumulative return: {cum_ret*100:.2f}%")
    if ally_cum is not None:
        print(f"  ALLY short return: {ally_cum*100:.2f}%")
    if cacc_cum is not None:
        print(f"  CACC short return: {cacc_cum*100:.2f}%")

if __name__ == "__main__":
    main()
