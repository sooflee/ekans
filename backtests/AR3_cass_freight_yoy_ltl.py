"""
AR-3: Cass Freight YoY Positive → Long ODFL/SAIA
When Cass Freight Index Shipments turns positive YoY after 12+ consecutive
negative months → long equal-weight {ODFL, SAIA} at month-end, hold 12 months.
FRED series: FRGSHPUSM649NCIS
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, load_fred, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AR-3"
FRED_SERIES = "FRGSHPUSM649NCIS"
LONG_BASKET = ["ODFL", "SAIA"]
HOLD_MONTHS = 12


def run():
    # --- Load Cass Freight data from FRED ---
    try:
        cass = load_fred(FRED_SERIES, start="1990-01-01")
        cass = cass[FRED_SERIES].dropna()
        print(f"Cass Freight data: {cass.index[0].date()} → {cass.index[-1].date()}, {len(cass)} obs")
    except Exception as e:
        print(f"FRED load failed: {e}")
        # Fallback: try anyway and mark failed if needed
        mark_failed(SIGNAL_ID, f"FRED data unavailable: {e}",
                     extra={"rule": "Cass YoY positive after 12+ neg months → long LTL",
                            "mechanism": "Freight cycle trough → LTL pricing power recovery",
                            "source": "FRED FRGSHPUSM649NCIS"})
        return

    # --- Compute YoY change ---
    cass_yoy = cass.pct_change(12)  # 12-month pct change
    cass_yoy = cass_yoy.dropna()
    print(f"YoY series: {cass_yoy.index[0].date()} → {cass_yoy.index[-1].date()}")

    # --- Identify triggers: first positive YoY after 12+ consecutive negative ---
    is_neg = (cass_yoy < 0).astype(int)
    triggers = []
    consec_neg = 0
    for i in range(len(cass_yoy)):
        if is_neg.iloc[i] == 1:
            consec_neg += 1
        else:
            if consec_neg >= 12:
                triggers.append(cass_yoy.index[i])
                print(f"  TRIGGER: {cass_yoy.index[i].date()} (after {consec_neg} neg months, YoY={cass_yoy.iloc[i]:.4f})")
            consec_neg = 0

    print(f"\nTotal triggers found: {len(triggers)}")
    if len(triggers) == 0:
        # Relax to 6+ consecutive negative months
        print("Relaxing to 6+ consecutive negative months...")
        consec_neg = 0
        for i in range(len(cass_yoy)):
            if is_neg.iloc[i] == 1:
                consec_neg += 1
            else:
                if consec_neg >= 6:
                    triggers.append(cass_yoy.index[i])
                    print(f"  TRIGGER (relaxed): {cass_yoy.index[i].date()} (after {consec_neg} neg months)")
                consec_neg = 0
        print(f"  Relaxed triggers: {len(triggers)}")

    if len(triggers) == 0:
        mark_failed(SIGNAL_ID, "No trigger months found in Cass data",
                     extra={"rule": "Cass YoY positive after 12+ neg months → long LTL",
                            "mechanism": "Freight cycle trough → LTL pricing power recovery",
                            "source": "FRED FRGSHPUSM649NCIS"})
        return

    # --- Load equity prices ---
    all_tickers = LONG_BASKET + ["SPY"]
    prices = load_prices(all_tickers, start="2000-01-01")
    rets = daily_returns(prices)
    print(f"Price data: {prices.index[0].date()} → {prices.index[-1].date()}")

    # --- Build trade windows ---
    trade_pnls = []
    trade_details = []
    for trig in triggers:
        entry = trig + pd.offsets.MonthEnd(0)  # end of trigger month
        exit_date = entry + pd.DateOffset(months=HOLD_MONTHS)

        # Find actual trading days
        mask = (rets.index >= entry) & (rets.index <= exit_date)
        window_rets = rets.loc[mask]

        avail = [t for t in LONG_BASKET if t in window_rets.columns and window_rets[t].notna().sum() > 20]
        if not avail:
            print(f"  {entry.date()}: no ticker data available, skipping")
            continue

        # Equal-weight long
        port_ret = window_rets[avail].mean(axis=1)
        bench_ret = window_rets["SPY"] if "SPY" in window_rets.columns else None

        cum = (1 + port_ret).cumprod()
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        spy_cum = (1 + bench_ret).cumprod() if bench_ret is not None else None
        spy_total = spy_cum.iloc[-1] - 1 if spy_cum is not None and len(spy_cum) > 0 else 0

        trade_details.append({
            "trigger": str(trig.date()),
            "entry": str(entry.date()),
            "exit": str(exit_date.date()),
            "tickers": avail,
            "total_return": float(total_ret),
            "spy_return": float(spy_total),
            "excess": float(total_ret - spy_total),
            "n_days": len(port_ret),
        })
        trade_pnls.append(port_ret)
        print(f"  {entry.date()}: {avail} → {total_ret*100:.1f}% (SPY {spy_total*100:.1f}%, excess {(total_ret-spy_total)*100:.1f}%)")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows with price data",
                     extra={"rule": "Cass YoY positive after 12+ neg months → long LTL",
                            "mechanism": "Freight cycle trough → LTL pricing power recovery",
                            "source": "FRED FRGSHPUSM649NCIS"})
        return

    # --- Combine all trade windows ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    # Remove duplicate dates (overlapping windows) - take mean
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="AR-3 Cass Freight YoY → Long LTL")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Cass Freight YoY positive after 12+ consecutive negative months → long ODFL+SAIA equal-weight 12mo",
        "mechanism": "Freight cycle trough signals supply rationalization → surviving LTL carriers gain pricing power",
        "source": "FRED FRGSHPUSM649NCIS + yfinance",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
