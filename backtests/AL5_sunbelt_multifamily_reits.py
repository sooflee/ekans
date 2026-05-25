"""
AL-5  Sunbelt Multifamily REITs
Regime trade: long equal-weight {CPT, MAA, INVH} from Jan 2022 through present.
Compare to VNQ (broad REIT ETF) and SPY.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AL-5"

def main():
    try:
        long_tickers = ["CPT", "MAA", "INVH"]
        bench_tickers = ["VNQ", "SPY"]
        all_tickers = long_tickers + bench_tickers

        px = load_prices(all_tickers, start="2021-06-01")
        ret = daily_returns(px)

        avail = [t for t in long_tickers if t in ret.columns and ret[t].dropna().shape[0] > 50]
        if not avail:
            mark_failed(SIGNAL_ID, "No Sunbelt REIT tickers available",
                        extra={"rule": "Long CPT/MAA/INVH from Jan 2022",
                               "mechanism": "Sunbelt population migration supports multifamily demand",
                               "source": "AL-5 signal catalog"})
            print("FAILED: no data")
            return

        print(f"Available: {avail}")

        start_date = pd.Timestamp("2022-01-03")
        mask = ret.index >= start_date
        ret_window = ret.loc[mask]

        basket = ret_window[avail].mean(axis=1).dropna()
        spy_ret = ret_window["SPY"].reindex(basket.index) if "SPY" in ret_window.columns else None
        vnq_ret = ret_window["VNQ"].reindex(basket.index) if "VNQ" in ret_window.columns else None

        # Basket metrics
        metrics = compute_metrics(basket, benchmark=spy_ret, name="AL-5 Sunbelt Multifamily REITs")
        print_metrics(metrics)

        # VNQ comparison
        if vnq_ret is not None and vnq_ret.dropna().shape[0] > 30:
            vnq_metrics = compute_metrics(vnq_ret.dropna(), benchmark=spy_ret, name="VNQ (broad REITs)")
            print()
            print_metrics(vnq_metrics)
            metrics["vnq_cagr"] = vnq_metrics.get("cagr")
            metrics["vnq_sharpe"] = vnq_metrics.get("sharpe")
            metrics["excess_vs_vnq"] = metrics.get("cagr", 0) - vnq_metrics.get("cagr", 0)

        metrics["tickers_used"] = avail
        metrics["status"] = "ok"

        save_result(SIGNAL_ID, metrics, extra={
            "rule": f"Long equal-weight {{{', '.join(avail)}}} from Jan 2022 through present",
            "mechanism": "Sunbelt states (TX, FL, NC, AZ, TN) saw massive population inflows post-COVID. Multifamily REITs with Sunbelt-focused portfolios (CPT, MAA) and single-family rental (INVH) benefit from demand tailwind vs coastal/broad REITs.",
            "source": "AL-5 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long Sunbelt multifamily REITs from Jan 2022",
                           "mechanism": "Sunbelt population migration",
                           "source": "AL-5 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
