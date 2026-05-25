"""
AR-1: FMCSA Carrier Exit Reversal → Long ODFL/SAIA
Proxy: hand-coded freight recession trough dates (supply-side cycle bottoms).
At each trough → long ODFL+SAIA equal-weight, hold 12 months.
Also tries BLS trucking employment (FRED CES4348400001) YoY as secondary trigger.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, load_fred, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AR-1"
LONG_BASKET = ["ODFL", "SAIA"]
HOLD_MONTHS = 12

# Known freight recession trough dates
TROUGH_DATES = [
    ("2009-03-01", "GFC trough"),
    ("2016-03-01", "Industrial recession trough"),
    ("2019-09-01", "Freight recession trough"),
    ("2024-10-01", "Current cycle trough estimate"),
]


def run():
    print("=== AR-1: FMCSA Carrier Exit Reversal (proxy: freight cycle troughs) ===\n")

    # --- Try BLS trucking employment as secondary signal ---
    try:
        emp = load_fred("CES4348400001", start="1990-01-01")
        emp = emp["CES4348400001"].dropna()
        emp_yoy = emp.pct_change(12).dropna()
        print(f"BLS trucking employment: {emp.index[0].date()} → {emp.index[-1].date()}")

        # Find months where YoY turns positive after decline
        neg_to_pos = []
        for i in range(1, len(emp_yoy)):
            if emp_yoy.iloc[i] > 0 and emp_yoy.iloc[i-1] <= 0:
                neg_to_pos.append(emp_yoy.index[i])
        print(f"BLS YoY sign flips (neg→pos): {len(neg_to_pos)}")
        for d in neg_to_pos:
            print(f"  {d.date()}: YoY = {emp_yoy.loc[d]:.4f}")
    except Exception as e:
        print(f"BLS trucking employment unavailable: {e}")

    # --- Primary: hand-coded trough dates ---
    print(f"\nHand-coded trough dates:")
    for d, desc in TROUGH_DATES:
        print(f"  {d}: {desc}")

    # --- Load equity prices ---
    all_tickers = LONG_BASKET + ["SPY"]
    prices = load_prices(all_tickers, start="2005-01-01")
    rets = daily_returns(prices)
    print(f"\nPrice data: {prices.index[0].date()} → {prices.index[-1].date()}")

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
            print(f"  {entry.date()}: no ticker data available, skipping")
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
        print(f"  {entry.date()} ({desc}): {avail} → {total_ret*100:.1f}% (SPY {spy_total*100:.1f}%, excess {(total_ret-spy_total)*100:.1f}%)")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows",
                     extra={"rule": "Freight cycle trough → long ODFL+SAIA 12mo",
                            "mechanism": "Carrier exit = supply destruction → pricing power for survivors",
                            "source": "Hand-coded trough dates + yfinance"})
        return

    # --- Combine all trade windows ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="AR-1 Carrier Exit → Long LTL")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Freight recession trough → long ODFL+SAIA equal-weight 12mo",
        "mechanism": "FMCSA carrier exits (supply destruction) → surviving LTL carriers gain pricing power and market share",
        "source": "Hand-coded trough dates (GFC, industrial, freight, COVID) + yfinance",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
