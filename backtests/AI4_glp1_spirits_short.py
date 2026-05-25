"""
AI-4: GLP-1 Adoption → Short Spirits (BF-B + STZ)

Regime trade: short BF-B + STZ equal-weight from Jun 2023 through present.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AI-4"

SHORT_TICKERS = ["BF-B", "STZ"]
BENCHMARK = "SPY"
START_DATE = "2023-06-01"

def main():
    all_tickers = SHORT_TICKERS + [BENCHMARK]
    try:
        prices = load_prices(all_tickers, start="2023-01-01")
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
    short_pnl = -short_basket_ret  # profit when stocks go down
    short_pnl = short_pnl.dropna()

    if len(short_pnl) < 30:
        mark_failed(SIGNAL_ID, "Insufficient data")
        print(f"FAILED {SIGNAL_ID}: insufficient data")
        return

    bench_ret = rets[BENCHMARK].reindex(short_pnl.index).dropna() if BENCHMARK in rets.columns else None
    metrics = compute_metrics(short_pnl, benchmark=bench_ret, name="GLP-1 → Short BF-B+STZ")

    # Cumulative return of the short
    cum_ret = (1 + short_pnl).cumprod().iloc[-1] - 1
    # Individual stocks
    bfb_cum = -((1 + rets["BF-B"]).cumprod().iloc[-1] - 1) if "BF-B" in rets.columns else None
    stz_cum = -((1 + rets["STZ"]).cumprod().iloc[-1] - 1) if "STZ" in rets.columns else None

    extra = {
        "status": "ok",
        "rule": "Short BF-B + STZ equal-weight from Jun 2023 (GLP-1 mass adoption inflection)",
        "mechanism": "GLP-1 drugs reduce alcohol cravings/consumption → spirits volume decline → revenue headwind for distillers",
        "source": "STEP trial data, pharmacy fill data, Nielsen scanner data showing spirits volume deceleration",
        "trade_type": "regime_short",
        "start_date": START_DATE,
        "short_basket_cum_return": float(cum_ret),
        "BF-B_short_return": float(bfb_cum) if bfb_cum is not None else None,
        "STZ_short_return": float(stz_cum) if stz_cum is not None else None,
    }

    save_result(SIGNAL_ID, metrics, extra)
    print_metrics(metrics)
    print(f"\nShort basket cumulative return: {cum_ret*100:.2f}%")
    if bfb_cum is not None:
        print(f"  BF-B short return: {bfb_cum*100:.2f}%")
    if stz_cum is not None:
        print(f"  STZ short return: {stz_cum*100:.2f}%")

if __name__ == "__main__":
    main()
