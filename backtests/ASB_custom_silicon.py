"""
AS-B: Inference Cost Reversal → Long Custom Silicon (MRVL/CRDO)
Regime trade: long equal-weight {MRVL, CRDO} from Jan 2024 (custom silicon
acceleration visible in hyperscaler capex) through present.
Benchmark: SPY and SOXX.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AS-B"
ENTRY_DATE = "2024-01-02"
LONG_BASKET = ["MRVL", "CRDO"]
BENCHMARKS = ["SPY", "SOXX"]


def run():
    print("=== AS-B: Inference Cost Reversal → Long Custom Silicon ===\n")
    print(f"Entry: {ENTRY_DATE}")
    print(f"Long: {LONG_BASKET} (equal-weight)")
    print(f"Benchmarks: {BENCHMARKS}\n")

    # --- Load prices ---
    all_tickers = LONG_BASKET + BENCHMARKS
    prices = load_prices(all_tickers, start="2023-01-01")
    rets = daily_returns(prices)

    entry = pd.Timestamp(ENTRY_DATE)
    mask = rets.index >= entry
    window_rets = rets.loc[mask]

    if len(window_rets) < 30:
        mark_failed(SIGNAL_ID, "Insufficient data after entry date",
                     extra={"rule": "Long MRVL+CRDO from Jan 2024",
                            "mechanism": "Custom silicon demand from hyperscaler inference buildout",
                            "source": "yfinance"})
        return

    avail = [t for t in LONG_BASKET if t in window_rets.columns and window_rets[t].notna().sum() > 20]
    if not avail:
        mark_failed(SIGNAL_ID, "No ticker data available",
                     extra={"rule": "Long MRVL+CRDO from Jan 2024",
                            "mechanism": "Custom silicon demand from hyperscaler inference buildout",
                            "source": "yfinance"})
        return

    # Equal-weight portfolio
    port_ret = window_rets[avail].mean(axis=1)

    # Individual returns
    print(f"Period: {window_rets.index[0].date()} → {window_rets.index[-1].date()} ({len(window_rets)} days)")
    for t in avail + BENCHMARKS:
        if t in window_rets.columns:
            cum = (1 + window_rets[t].fillna(0)).cumprod()
            print(f"  {t}: {(cum.iloc[-1]-1)*100:+.1f}%")

    port_cum = (1 + port_ret).cumprod()
    print(f"\n  Portfolio ({'+'.join(avail)}): {(port_cum.iloc[-1]-1)*100:+.1f}%")

    # Metrics vs SPY
    spy_rets = rets["SPY"].reindex(port_ret.index) if "SPY" in rets.columns else None
    metrics = compute_metrics(port_ret, benchmark=spy_rets, name="AS-B Custom Silicon")
    print_metrics(metrics)

    # Also compute vs SOXX
    if "SOXX" in rets.columns:
        soxx_rets = rets["SOXX"].reindex(port_ret.index)
        soxx_metrics = compute_metrics(port_ret, benchmark=soxx_rets, name="AS-B vs SOXX")
        print(f"\n  vs SOXX excess CAGR: {soxx_metrics.get('excess_cagr', 0)*100:+.1f}%")

    trade_detail = {
        "entry": ENTRY_DATE,
        "exit": str(window_rets.index[-1].date()),
        "tickers": avail,
        "portfolio_total_return": float(port_cum.iloc[-1] - 1),
        "n_days": len(window_rets),
    }
    for t in avail + BENCHMARKS:
        if t in window_rets.columns:
            cum = (1 + window_rets[t].fillna(0)).cumprod()
            trade_detail[f"{t}_total_return"] = float(cum.iloc[-1] - 1)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Long MRVL+CRDO equal-weight from Jan 2024 (regime trade)",
        "mechanism": "Hyperscaler capex shift to custom ASICs for inference → Marvell/Credo benefit from custom silicon + optical interconnect demand",
        "source": "yfinance",
        "trades": [trade_detail],
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
