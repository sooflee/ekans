"""
AL-4  East Asia Birth Collapse -> Pet vs Formula Pair
Regime trade: long (IDXX + ZTS + CHWY)/3 vs short (DANOY + RBGLY)/2
from Jan 2022 through present.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AL-4"

def main():
    try:
        long_tickers = ["IDXX", "ZTS", "CHWY"]
        short_tickers = ["DANOY", "RBGLY"]
        all_tickers = long_tickers + short_tickers + ["SPY"]

        px = load_prices(all_tickers, start="2021-06-01")
        ret = daily_returns(px)

        # Check which tickers are available
        avail_long = [t for t in long_tickers if t in ret.columns and ret[t].dropna().shape[0] > 50]
        avail_short = [t for t in short_tickers if t in ret.columns and ret[t].dropna().shape[0] > 50]

        print(f"Long leg available: {avail_long}")
        print(f"Short leg available: {avail_short}")

        if not avail_long:
            mark_failed(SIGNAL_ID, "No long-leg tickers available",
                        extra={"rule": "Long pet basket vs short formula basket from Jan 2022",
                               "mechanism": "Structural birth decline in East Asia shifts spending from baby to pet products",
                               "source": "AL-4 signal catalog"})
            print("FAILED: no long-leg data")
            return

        start_date = pd.Timestamp("2022-01-03")
        mask = ret.index >= start_date
        ret_window = ret.loc[mask]

        # Long leg: equal-weight
        long_ret = ret_window[avail_long].mean(axis=1)

        # Short leg (if available)
        if avail_short:
            short_ret = ret_window[avail_short].mean(axis=1)
            pair_pnl = long_ret - short_ret
            trade_desc = f"Long ({'+'.join(avail_long)})/3 vs Short ({'+'.join(avail_short)})/2"
        else:
            pair_pnl = long_ret
            trade_desc = f"Long-only ({'+'.join(avail_long)})/3 (short leg ADRs unavailable)"
            print("NOTE: Short-leg OTC ADRs not available, testing long-only pet basket")

        pair_pnl = pair_pnl.dropna()
        spy_ret = ret_window["SPY"].reindex(pair_pnl.index)

        metrics = compute_metrics(pair_pnl, benchmark=spy_ret, name="AL-4 Birth Collapse Pet Pair")
        print_metrics(metrics)

        # Also compute long-only metrics for reference
        long_metrics = compute_metrics(long_ret.dropna(), benchmark=spy_ret, name="Long Pet Basket Only")
        print()
        print_metrics(long_metrics)

        metrics["trade_description"] = trade_desc
        metrics["long_tickers_used"] = avail_long
        metrics["short_tickers_used"] = avail_short
        metrics["status"] = "ok"

        save_result(SIGNAL_ID, metrics, extra={
            "rule": trade_desc,
            "mechanism": "Structural birth decline in East Asia (China births fell from 17.9M in 2016 to ~9M in 2023) shifts consumer spending from baby formula to pet products. Pet ownership rises as substitute for child-rearing.",
            "source": "AL-4 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long pet vs short formula from Jan 2022",
                           "mechanism": "Birth collapse shifts spending to pets",
                           "source": "AL-4 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
