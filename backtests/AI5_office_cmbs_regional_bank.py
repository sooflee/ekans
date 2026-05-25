"""
AI-5: Office CMBS Distress → Regional Bank Pair Trade

Regime trade: short OZK + NYCB vs long JPM + WFC from Jan 2023 through present.
Expression: (JPM+WFC)/2 - (OZK+NYCB)/2 spread.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AI-5"

LONG_TICKERS = ["JPM", "WFC"]
SHORT_TICKERS = ["OZK", "FLG"]  # NYCB renamed to FLG (Flagstar Financial) in 2024
START_DATE = "2023-01-03"

def main():
    all_tickers = LONG_TICKERS + SHORT_TICKERS + ["SPY"]
    try:
        prices = load_prices(all_tickers, start="2022-06-01")
    except Exception as e:
        mark_failed(SIGNAL_ID, f"Data load failed: {e}")
        print(f"FAILED {SIGNAL_ID}: {e}")
        return

    if prices.empty:
        mark_failed(SIGNAL_ID, "No price data")
        print(f"FAILED {SIGNAL_ID}: no data")
        return

    # Check we have all tickers
    missing = [t for t in LONG_TICKERS + SHORT_TICKERS if t not in prices.columns]
    if missing:
        mark_failed(SIGNAL_ID, f"Missing tickers: {missing}")
        print(f"FAILED {SIGNAL_ID}: missing {missing}")
        return

    rets = daily_returns(prices)
    rets = rets[rets.index >= START_DATE]

    # Pair spread: equal-weight long big banks, short regional/CRE-exposed banks
    # Daily PnL = 0.5*(JPM_ret + WFC_ret) - 0.5*(OZK_ret + NYCB_ret)
    long_ret = rets[LONG_TICKERS].mean(axis=1)
    short_ret = rets[SHORT_TICKERS].mean(axis=1)
    spread_pnl = long_ret - short_ret
    spread_pnl = spread_pnl.dropna()

    if len(spread_pnl) < 30:
        mark_failed(SIGNAL_ID, "Insufficient data after start date")
        print(f"FAILED {SIGNAL_ID}: insufficient data")
        return

    bench_ret = rets["SPY"].reindex(spread_pnl.index).dropna() if "SPY" in rets.columns else None
    metrics = compute_metrics(spread_pnl, benchmark=bench_ret, name="Office CMBS Pair: Long JPM+WFC / Short OZK+NYCB")

    # Also compute individual leg performance
    long_cum = (1 + long_ret).cumprod().iloc[-1] - 1
    short_cum = (1 + short_ret).cumprod().iloc[-1] - 1

    extra = {
        "status": "ok",
        "rule": "Short OZK+NYCB (CRE-exposed regionals) vs Long JPM+WFC from Jan 2023 when office vacancy + CMBS DQ structural",
        "mechanism": "Remote work → structural office vacancy → CMBS delinquencies hit regional banks with concentrated CRE exposure",
        "source": "Trepp CMBS DQ data, CBRE vacancy reports, bank 10-K CRE exposure disclosures",
        "trade_type": "regime_pair",
        "start_date": START_DATE,
        "long_leg_cum_return": float(long_cum),
        "short_leg_cum_return": float(short_cum),
        "spread_cum_return": float((1 + spread_pnl).cumprod().iloc[-1] - 1),
    }

    save_result(SIGNAL_ID, metrics, extra)
    print_metrics(metrics)
    print(f"\nLong leg (JPM+WFC) cumulative: {long_cum*100:.2f}%")
    print(f"Short leg (OZK+NYCB) cumulative: {short_cum*100:.2f}%")
    print(f"Spread cumulative: {((1+spread_pnl).cumprod().iloc[-1]-1)*100:.2f}%")

if __name__ == "__main__":
    main()
