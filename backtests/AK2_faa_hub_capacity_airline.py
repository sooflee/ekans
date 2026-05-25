"""
AK-2  FAA Hub Capacity -> Airline Pricing
Hand-coded FAA capacity reduction events:
  2023-01-11 FAA NOTAM system failure (nationwide ground stop)
  2025-11-15 FAA ORD/SFO capacity reduction announcement (approximate)
  2026-03-15 SFO runway closure effective date (approximate)
For each: long the dominant carrier (UAL for SFO/ORD, DAL for ATL) from event date, hold 60 days.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AK-2"

# (event_date, ticker, description)
FAA_EVENTS = [
    (pd.Timestamp("2023-01-11"), "UAL", "FAA NOTAM system failure - nationwide ground stop"),
    (pd.Timestamp("2025-11-15"), "UAL", "FAA ORD/SFO capacity reduction announcement"),
    (pd.Timestamp("2026-03-15"), "UAL", "SFO runway closure effective date"),
]
HOLD_DAYS = 60

def main():
    try:
        tickers = ["UAL", "DAL", "SPY"]
        px = load_prices(tickers, start="2022-01-01")
        ret = daily_returns(px)

        avail = [t for t in ["UAL", "DAL"] if t in ret.columns and ret[t].dropna().shape[0] > 50]
        if not avail:
            mark_failed(SIGNAL_ID, "No airline tickers available",
                        extra={"rule": "Long dominant carrier after FAA capacity events",
                               "mechanism": "Capacity constraints -> pricing power for incumbents",
                               "source": "AK-2 signal catalog"})
            print("FAILED: no airline data")
            return

        print(f"Available airline tickers: {avail}")

        event_results = []
        event_pnls = []
        spy_pnls = []

        for evt_date, ticker, desc in FAA_EVENTS:
            if ticker not in avail:
                print(f"  {evt_date.date()} ({desc}): {ticker} not available, skipping")
                continue

            mask = ret.index >= evt_date
            if mask.sum() < HOLD_DAYS:
                # Try with available data
                if mask.sum() < 5:
                    print(f"  {evt_date.date()} ({desc}): insufficient data ({mask.sum()} days)")
                    continue

            trade_start = ret.index[mask][0]
            start_loc = ret.index.get_loc(trade_start)
            end_loc = min(start_loc + HOLD_DAYS, len(ret) - 1)

            airline_slice = ret[ticker].iloc[start_loc:end_loc + 1]
            spy_slice = ret["SPY"].iloc[start_loc:end_loc + 1] if "SPY" in ret.columns else None

            eq = (1 + airline_slice).cumprod()
            total_ret = eq.iloc[-1] - 1

            spy_ret_total = None
            if spy_slice is not None and len(spy_slice) > 0:
                spy_eq = (1 + spy_slice).cumprod()
                spy_ret_total = spy_eq.iloc[-1] - 1

            n_held = len(airline_slice)
            event_results.append({
                "event_date": str(evt_date.date()),
                "ticker": ticker,
                "description": desc,
                "trade_start": str(airline_slice.index[0].date()),
                "trade_end": str(airline_slice.index[-1].date()),
                "airline_return": float(total_ret),
                "spy_return": float(spy_ret_total) if spy_ret_total is not None else None,
                "excess": float(total_ret - spy_ret_total) if spy_ret_total is not None else None,
                "n_days": n_held
            })
            event_pnls.append(airline_slice)
            if spy_slice is not None:
                spy_pnls.append(spy_slice)

            spy_str = f"  SPY={spy_ret_total*100:.2f}%" if spy_ret_total is not None else ""
            print(f"  {evt_date.date()} ({ticker}, {desc}): {total_ret*100:.2f}%{spy_str} ({n_held}d)")

        if not event_results:
            mark_failed(SIGNAL_ID, "No event trades could be executed",
                        extra={"rule": "Long dominant carrier after FAA capacity events",
                               "mechanism": "Capacity constraints -> pricing power",
                               "source": "AK-2 signal catalog"})
            return

        avg_airline = np.mean([e["airline_return"] for e in event_results])
        avg_spy = np.mean([e["spy_return"] for e in event_results if e["spy_return"] is not None])
        avg_excess = np.mean([e["excess"] for e in event_results if e["excess"] is not None])

        print(f"\nAvg airline 60d return: {avg_airline*100:.2f}%")
        if not np.isnan(avg_spy):
            print(f"Avg SPY 60d return: {avg_spy*100:.2f}%")
            print(f"Avg excess: {avg_excess*100:.2f}%")

        # Combined metrics
        if event_pnls:
            combined = pd.concat(event_pnls)
            combined_spy = pd.concat(spy_pnls) if spy_pnls else None
            metrics = compute_metrics(combined, benchmark=combined_spy, name="AK-2 FAA Hub Capacity Airline")
            print()
            print_metrics(metrics)
        else:
            metrics = {"name": "AK-2 FAA Hub Capacity Airline", "n_days": 0}

        metrics["event_trades"] = event_results
        metrics["avg_airline_return_60d"] = float(avg_airline)
        metrics["avg_spy_return_60d"] = float(avg_spy) if not np.isnan(avg_spy) else None
        metrics["avg_excess_60d"] = float(avg_excess) if not np.isnan(avg_excess) else None
        metrics["n_events"] = len(event_results)
        metrics["tickers_used"] = list(set(e["ticker"] for e in event_results))
        metrics["status"] = "ok"
        metrics["caveat"] = f"N={len(event_results)} events. Mixed event types (outage vs capacity order vs runway closure). Small sample."

        save_result(SIGNAL_ID, metrics, extra={
            "rule": "Long dominant carrier (UAL for SFO/ORD) for 60 days after FAA capacity reduction events",
            "mechanism": "FAA capacity constraints at major hubs reduce available slots, benefiting dominant carriers who can reprice limited capacity. Hub dominance (UAL at SFO/ORD, DAL at ATL) means pricing power concentrates with the incumbent.",
            "source": "AK-2 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long dominant carrier after FAA events",
                           "mechanism": "Hub capacity constraints -> airline pricing power",
                           "source": "AK-2 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
