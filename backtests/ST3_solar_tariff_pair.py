"""
ST-3: Solar Module Price / Tariff Events → Long FSLR (+FLNC) / Short JKS+CSIQ
Hand-coded major solar tariff events where module prices spiked.
For each: long domestic (FSLR, FLNC post-2021), short Chinese (JKS, CSIQ), 6 months.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "ST-3"
HOLD_MONTHS = 6

# (event_date, description, long_tickers, short_tickers)
TARIFF_EVENTS = [
    ("2018-01-22", "Trump Sec 201 tariff (30% on imported cells)", ["FSLR"], ["JKS", "CSIQ"]),
    ("2022-03-28", "Commerce Dept AD/CVD investigation on SE Asian cells", ["FSLR", "FLNC"], ["JKS", "CSIQ"]),
    ("2024-06-06", "Commerce Dept preliminary AD/CVD determination", ["FSLR", "FLNC"], ["JKS", "CSIQ"]),
    ("2025-04-22", "Commerce Dept final AD/CVD (81-3404% rates)", ["FSLR", "FLNC"], ["JKS", "CSIQ"]),
]


def run():
    print("=== ST-3: Solar Tariff → Long FSLR/FLNC, Short JKS/CSIQ ===\n")

    for d, desc, longs, shorts in TARIFF_EVENTS:
        print(f"  {d}: {desc}")
        print(f"    Long: {longs}, Short: {shorts}")

    # --- Load all needed tickers ---
    all_tickers = list(set(["FSLR", "FLNC", "JKS", "CSIQ", "SPY"]))
    prices = load_prices(all_tickers, start="2010-01-01")
    rets = daily_returns(prices)
    print(f"\nPrice data: {prices.index[0].date()} → {prices.index[-1].date()}")
    for t in all_tickers:
        if t in prices.columns:
            first = prices[t].dropna().index[0].date()
            print(f"  {t}: from {first}")

    # --- Build trade windows ---
    trade_pnls = []
    trade_details = []
    for event_date_str, desc, long_tickers, short_tickers in TARIFF_EVENTS:
        entry = pd.Timestamp(event_date_str)
        exit_date = entry + pd.DateOffset(months=HOLD_MONTHS)

        mask = (rets.index >= entry) & (rets.index <= exit_date)
        window_rets = rets.loc[mask]

        if len(window_rets) < 10:
            print(f"\n  {entry.date()}: insufficient data ({len(window_rets)} days), skipping")
            continue

        # Available tickers
        avail_long = [t for t in long_tickers if t in window_rets.columns and window_rets[t].notna().sum() > 10]
        avail_short = [t for t in short_tickers if t in window_rets.columns and window_rets[t].notna().sum() > 10]

        if not avail_long or not avail_short:
            print(f"\n  {entry.date()}: missing tickers (long={avail_long}, short={avail_short}), skipping")
            continue

        # Equal-weight: long side + short side, 4 legs total
        n_legs = len(avail_long) + len(avail_short)
        wt = 1.0 / n_legs

        port_ret = pd.Series(0.0, index=window_rets.index)
        for t in avail_long:
            port_ret += wt * window_rets[t].fillna(0)
        for t in avail_short:
            port_ret -= wt * window_rets[t].fillna(0)  # short = negative

        bench_ret = window_rets["SPY"] if "SPY" in window_rets.columns else None

        cum = (1 + port_ret).cumprod()
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        spy_cum = (1 + bench_ret).cumprod() if bench_ret is not None else None
        spy_total = spy_cum.iloc[-1] - 1 if spy_cum is not None and len(spy_cum) > 0 else 0

        # Separate long/short legs
        long_ret = (1 + window_rets[avail_long].mean(axis=1)).cumprod().iloc[-1] - 1
        short_ret = (1 + window_rets[avail_short].mean(axis=1)).cumprod().iloc[-1] - 1

        trade_details.append({
            "event_date": event_date_str,
            "description": desc,
            "long": avail_long,
            "short": avail_short,
            "total_return": float(total_ret),
            "long_leg_return": float(long_ret),
            "short_leg_return": float(short_ret),
            "spy_return": float(spy_total),
            "n_days": len(port_ret),
        })
        trade_pnls.append(port_ret)
        print(f"\n  {entry.date()} ({desc[:40]}...):")
        print(f"    Long {avail_long}: {long_ret*100:+.1f}%")
        print(f"    Short {avail_short}: {-short_ret*100:+.1f}% (P&L from short)")
        print(f"    Net: {total_ret*100:+.1f}% | SPY: {spy_total*100:+.1f}%")

    if not trade_pnls:
        mark_failed(SIGNAL_ID, "No valid trade windows",
                     extra={"rule": "Solar tariff event → long FSLR/FLNC, short JKS/CSIQ 6mo",
                            "mechanism": "Tariff = module price spike → benefits domestic mfg, hurts Chinese exporters",
                            "source": "Hand-coded tariff dates + yfinance"})
        return

    # --- Combine ---
    all_pnl = pd.concat(trade_pnls).sort_index()
    all_pnl = all_pnl.groupby(all_pnl.index).mean()

    spy_rets = rets["SPY"].reindex(all_pnl.index)
    metrics = compute_metrics(all_pnl, benchmark=spy_rets, name="ST-3 Solar Tariff Pair")
    print_metrics(metrics)

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Solar tariff event → long FSLR(+FLNC), short JKS+CSIQ, equal-weight 4 legs, hold 6mo",
        "mechanism": "Import tariffs spike module prices → domestic manufacturers benefit, Chinese exporters lose US market share",
        "source": "Hand-coded tariff dates + yfinance",
        "trades": trade_details,
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
