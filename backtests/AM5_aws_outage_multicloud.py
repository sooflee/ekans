"""
AM-5  AWS Outage -> Multi-Cloud Basket
Hand-coded major AWS outage events (6+ hours):
  2021-12-07 (us-east-1, ~7 hours)
  2023-06-13 (Lambda/CloudFront outage)
  2026-03-15 (UAE datacenter fire -- approximate)
For each: long (DDOG + NET + FSLY)/3 from next trading day, hold 30 days.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AM-5"

AWS_OUTAGE_EVENTS = [
    pd.Timestamp("2021-12-07"),  # us-east-1 outage
    pd.Timestamp("2023-06-13"),  # Lambda/CloudFront outage
    pd.Timestamp("2026-03-15"),  # UAE datacenter fire (estimated)
]
HOLD_DAYS = 30

def main():
    try:
        tickers = ["DDOG", "NET", "FSLY", "SPY"]
        px = load_prices(tickers, start="2021-01-01")
        ret = daily_returns(px)

        basket_tickers = ["DDOG", "NET", "FSLY"]
        avail = [t for t in basket_tickers if t in ret.columns and ret[t].dropna().shape[0] > 50]

        if not avail:
            mark_failed(SIGNAL_ID, "No multi-cloud tickers available",
                        extra={"rule": "Long DDOG/NET/FSLY after AWS outages",
                               "mechanism": "AWS outages accelerate multi-cloud adoption narrative",
                               "source": "AM-5 signal catalog"})
            print("FAILED: no data")
            return

        print(f"Available basket tickers: {avail}")

        event_results = []
        event_pnls = []
        spy_pnls = []

        for evt in AWS_OUTAGE_EVENTS:
            # Enter on next trading day after event
            next_day = evt + pd.Timedelta(days=1)
            mask = ret.index >= next_day
            if mask.sum() < HOLD_DAYS:
                print(f"  {evt.date()}: insufficient data after event ({mask.sum()} days available)")
                continue

            trade_start = ret.index[mask][0]
            start_loc = ret.index.get_loc(trade_start)
            end_loc = min(start_loc + HOLD_DAYS, len(ret) - 1)

            basket_slice = ret[avail].iloc[start_loc:end_loc + 1].mean(axis=1)
            spy_slice = ret["SPY"].iloc[start_loc:end_loc + 1] if "SPY" in ret.columns else None

            eq = (1 + basket_slice).cumprod()
            total_ret = eq.iloc[-1] - 1

            spy_ret_total = None
            if spy_slice is not None and len(spy_slice) > 0:
                spy_eq = (1 + spy_slice).cumprod()
                spy_ret_total = spy_eq.iloc[-1] - 1

            event_results.append({
                "event_date": str(evt.date()),
                "trade_start": str(basket_slice.index[0].date()),
                "trade_end": str(basket_slice.index[-1].date()),
                "basket_return_30d": float(total_ret),
                "spy_return_30d": float(spy_ret_total) if spy_ret_total is not None else None,
                "excess": float(total_ret - spy_ret_total) if spy_ret_total is not None else None,
                "n_days": len(basket_slice)
            })
            event_pnls.append(basket_slice)
            if spy_slice is not None:
                spy_pnls.append(spy_slice)

            excess_str = f"  excess={((total_ret - spy_ret_total)*100):.2f}%" if spy_ret_total is not None else ""
            print(f"  {evt.date()}: basket={total_ret*100:.2f}%  SPY={spy_ret_total*100:.2f}%{excess_str} ({len(basket_slice)} days)")

        if not event_results:
            mark_failed(SIGNAL_ID, "No event trades could be executed",
                        extra={"rule": "Long multi-cloud basket after AWS outages",
                               "mechanism": "AWS outages accelerate multi-cloud adoption",
                               "source": "AM-5 signal catalog"})
            return

        avg_basket = np.mean([e["basket_return_30d"] for e in event_results])
        avg_spy = np.mean([e["spy_return_30d"] for e in event_results if e["spy_return_30d"] is not None])
        avg_excess = np.mean([e["excess"] for e in event_results if e["excess"] is not None])

        print(f"\nAvg basket 30d return: {avg_basket*100:.2f}%")
        print(f"Avg SPY 30d return: {avg_spy*100:.2f}%")
        print(f"Avg excess: {avg_excess*100:.2f}%")

        # Combined metrics
        if event_pnls:
            combined = pd.concat(event_pnls)
            combined_spy = pd.concat(spy_pnls) if spy_pnls else None
            metrics = compute_metrics(combined, benchmark=combined_spy, name="AM-5 AWS Outage Multi-Cloud")
            print()
            print_metrics(metrics)
        else:
            metrics = {"name": "AM-5 AWS Outage Multi-Cloud", "n_days": 0}

        metrics["event_trades"] = event_results
        metrics["avg_basket_return_30d"] = float(avg_basket)
        metrics["avg_spy_return_30d"] = float(avg_spy)
        metrics["avg_excess_30d"] = float(avg_excess)
        metrics["n_events"] = len(event_results)
        metrics["tickers_used"] = avail
        metrics["status"] = "ok"
        metrics["caveat"] = f"N={len(event_results)} events. Small sample. Also, post-outage moves may reflect broader tech sentiment, not outage-specific alpha."

        save_result(SIGNAL_ID, metrics, extra={
            "rule": "Long (DDOG+NET+FSLY)/3 for 30 days after major AWS outages (6+ hours)",
            "mechanism": "Major AWS outages (us-east-1 2021, Lambda 2023, UAE fire 2026) accelerate enterprise multi-cloud adoption narrative, benefiting monitoring (DDOG), edge/CDN (NET, FSLY) providers as alternatives to AWS-only architectures.",
            "source": "AM-5 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long multi-cloud after AWS outages",
                           "mechanism": "AWS outages -> multi-cloud adoption",
                           "source": "AM-5 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
