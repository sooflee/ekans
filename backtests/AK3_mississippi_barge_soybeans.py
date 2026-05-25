"""
AK-3  Mississippi Barge Rate -> Long Soybeans
Hand-coded years when Mississippi River low water coincided with high barge rates (Sep-Nov):
  2022-10-01, 2023-09-15, 2024-10-01
For each: long ZS=F from event date, hold 30 days.
Compare to buying ZS=F randomly in Sep-Nov of non-drought years.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AK-3"

# Drought/low-water events with high barge rates
DROUGHT_EVENTS = [
    pd.Timestamp("2022-10-03"),  # Historic low at Memphis
    pd.Timestamp("2023-09-15"),  # Repeat low water
    pd.Timestamp("2024-10-01"),  # Drought recurrence
]

# Non-drought comparison years (Sep-Nov random entry)
NON_DROUGHT_YEARS = [2019, 2020, 2021]
HOLD_DAYS = 30

def main():
    try:
        px = load_prices(["ZS=F"], start="2018-01-01")
        if px.empty or "ZS=F" not in px.columns:
            # Try alternative soybean ticker
            px = load_prices(["SOYB"], start="2018-01-01")
            if px.empty:
                mark_failed(SIGNAL_ID, "No soybean price data (ZS=F or SOYB)",
                            extra={"rule": "Long soybeans on Mississippi low-water events",
                                   "mechanism": "Low water -> high barge rates -> export bottleneck -> soy price spike",
                                   "source": "AK-3 signal catalog"})
                print("FAILED: no soybean data")
                return
            ticker = "SOYB"
        else:
            ticker = "ZS=F"

        ret = daily_returns(px)
        print(f"Using ticker: {ticker}")
        print(f"Data range: {ret.index[0].date()} to {ret.index[-1].date()}")

        # Event trades
        event_results = []
        event_pnls = []
        for evt in DROUGHT_EVENTS:
            # Find the next trading day on or after event
            mask = ret.index >= evt
            if mask.sum() < HOLD_DAYS:
                print(f"  {evt.date()}: insufficient data after event")
                continue

            trade_start = ret.index[mask][0]
            trade_end_idx = min(ret.index.get_loc(trade_start) + HOLD_DAYS, len(ret) - 1)
            trade_slice = ret[ticker].iloc[ret.index.get_loc(trade_start):trade_end_idx + 1]

            eq = (1 + trade_slice).cumprod()
            total_ret = eq.iloc[-1] - 1

            event_results.append({
                "event_date": str(evt.date()),
                "trade_start": str(trade_slice.index[0].date()),
                "trade_end": str(trade_slice.index[-1].date()),
                "return_30d": float(total_ret),
                "n_days": len(trade_slice)
            })
            event_pnls.append(trade_slice)
            print(f"  Drought {evt.date()}: {total_ret*100:.2f}% over {len(trade_slice)} days")

        # Non-drought comparison: buy Oct 1 in non-drought years, hold 30 days
        control_results = []
        for yr in NON_DROUGHT_YEARS:
            entry = pd.Timestamp(f"{yr}-10-01")
            mask = ret.index >= entry
            if mask.sum() < HOLD_DAYS:
                continue
            trade_start = ret.index[mask][0]
            trade_end_idx = min(ret.index.get_loc(trade_start) + HOLD_DAYS, len(ret) - 1)
            trade_slice = ret[ticker].iloc[ret.index.get_loc(trade_start):trade_end_idx + 1]

            eq = (1 + trade_slice).cumprod()
            total_ret = eq.iloc[-1] - 1
            control_results.append({
                "year": yr,
                "return_30d": float(total_ret),
                "n_days": len(trade_slice)
            })
            print(f"  Control {yr}: {total_ret*100:.2f}% over {len(trade_slice)} days")

        if not event_results:
            mark_failed(SIGNAL_ID, "No event trades could be executed",
                        extra={"rule": "Long soybeans on Mississippi low-water events",
                               "mechanism": "Low water -> high barge rates -> soy price spike",
                               "source": "AK-3 signal catalog"})
            return

        avg_event = np.mean([e["return_30d"] for e in event_results])
        avg_control = np.mean([c["return_30d"] for c in control_results]) if control_results else 0

        print(f"\nAvg drought event 30d return: {avg_event*100:.2f}%")
        print(f"Avg control 30d return: {avg_control*100:.2f}%")
        print(f"Edge: {(avg_event - avg_control)*100:.2f}%")

        # Concatenate event PnLs for metrics
        if event_pnls:
            combined = pd.concat(event_pnls)
            metrics = compute_metrics(combined, name="AK-3 Mississippi Barge Soybeans")
            print()
            print_metrics(metrics)
        else:
            metrics = {"name": "AK-3 Mississippi Barge Soybeans", "n_days": 0, "error": "no trades"}

        metrics["event_trades"] = event_results
        metrics["control_trades"] = control_results
        metrics["avg_event_return_30d"] = float(avg_event)
        metrics["avg_control_return_30d"] = float(avg_control)
        metrics["edge_vs_control"] = float(avg_event - avg_control)
        metrics["n_events"] = len(event_results)
        metrics["ticker_used"] = ticker
        metrics["status"] = "ok"
        metrics["caveat"] = f"Only N={len(event_results)} drought events. Small sample."

        save_result(SIGNAL_ID, metrics, extra={
            "rule": "Long soybeans (ZS=F) for 30 days after Mississippi River low-water/high-barge-rate events in Sep-Nov",
            "mechanism": "Mississippi River low water levels cause barge rate spikes (800%+ of tariff in 2022), creating export bottlenecks at Gulf ports. Soybean exports are concentrated in Sep-Nov harvest; disruption lifts prices.",
            "source": "AK-3 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long soybeans on Mississippi low-water",
                           "mechanism": "Barge rate spike -> soy price",
                           "source": "AK-3 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
