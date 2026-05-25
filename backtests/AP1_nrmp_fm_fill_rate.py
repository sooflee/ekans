"""
AP-1: NRMP FM Fill Rate → Long AMN / Short Hospital Basket
Hand-coded NRMP March release dates. Declining Family Medicine fill rate signals
worsening physician shortage → long staffing agency AMN, short hospitals.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AP-1"
HOLD_MONTHS = 12
HOSPITAL_BASKET = ["HCA", "THC", "UHS", "CYH"]

# (date, description, fm_fill_rate)
NRMP_EVENTS = [
    ("2024-03-15", "NRMP 2024 — FM fill rate 87.8%", 0.878),
    ("2025-03-14", "NRMP 2025 — FM fill rate 85.0%", 0.850),
    ("2026-03-13", "NRMP 2026 — FM fill rate 83.6%, emergency panel", 0.836),
]


def run():
    print("=== AP-1: NRMP FM Fill Rate → Long AMN / Short Hospitals ===\n")
    for d, desc, rate in NRMP_EVENTS:
        print(f"  {d}: {desc} (fill rate = {rate:.1%})")

    # --- Load prices ---
    all_tickers = ["AMN"] + HOSPITAL_BASKET + ["SPY"]
    prices = load_prices(all_tickers, start="2020-01-01")
    rets = daily_returns(prices)
    print(f"\nPrice data: {prices.index[0].date()} → {prices.index[-1].date()}")

    # --- Build trade windows ---
    trade_pnls = []
    trade_details = []
    for event_date_str, desc, fill_rate in NRMP_EVENTS:
        entry = pd.Timestamp(event_date_str)
        exit_date = entry + pd.DateOffset(months=HOLD_MONTHS)

        mask = (rets.index >= entry) & (rets.index <= exit_date)
        window_rets = rets.loc[mask]

        if len(window_rets) < 10:
            print(f"\n  {entry.date()}: insufficient data ({len(window_rets)} days), skipping")
            continue

        # Check AMN availability
        if "AMN" not in window_rets.columns or window_rets["AMN"].notna().sum() < 10:
            print(f"\n  {entry.date()}: AMN data unavailable, skipping")
            continue

        # Available hospital tickers
        avail_hosp = [t for t in HOSPITAL_BASKET if t in window_rets.columns and window_rets[t].notna().sum() > 10]
        if not avail_hosp:
            print(f"\n  {entry.date()}: no hospital data, skipping")
            continue

        # Long AMN (50%), Short hospital basket (50% split equally)
        hosp_ret = window_rets[avail_hosp].mean(axis=1)
        port_ret = 0.5 * window_rets["AMN"].fillna(0) - 0.5 * hosp_ret.fillna(0)

        amn_cum = (1 + window_rets["AMN"].fillna(0)).cumprod()
        hosp_cum = (1 + hosp_ret.fillna(0)).cumprod()
        net_cum = (1 + port_ret).cumprod()

        amn_total = amn_cum.iloc[-1] - 1
        hosp_total = hosp_cum.iloc[-1] - 1
        net_total = net_cum.iloc[-1] - 1

        trade_details.append({
            "event_date": event_date_str,
            "description": desc,
            "fm_fill_rate": fill_rate,
            "amn_return": float(amn_total),
            "hospital_basket_return": float(hosp_total),
            "net_return": float(net_total),
            "hospital_tickers": avail_hosp,
            "n_days": len(window_rets),
        })
        trade_pnls.append(port_ret)
        print(f"\n  {entry.date()} (FM fill {fill_rate:.1%}):")
        print(f"    AMN: {amn_total*100:+.1f}%")
        print(f"    Hospitals ({', '.join(avail_hosp)}): {hosp_total*100:+.1f}%")
        print(f"    Net L/S: {net_total*100:+.1f}%")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows",
                     extra={"rule": "NRMP FM fill rate decline → long AMN, short hospitals 12mo",
                            "mechanism": "Declining FM fill rate → physician shortage → hospitals need more locum tenens staffing",
                            "source": "Hand-coded NRMP dates + yfinance"})
        return

    # --- Combine ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="AP-1 NRMP FM Fill → Long AMN / Short Hospitals")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "NRMP FM fill rate release (March) → long AMN, short (HCA+THC+UHS+CYH)/4, hold 12mo",
        "mechanism": "Declining FM fill rate signals worsening physician shortage → hospitals forced to use more expensive locum staffing (AMN benefits)",
        "source": "Hand-coded NRMP dates + yfinance",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
