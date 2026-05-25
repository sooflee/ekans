"""
AM-1  Hormuz Refiner Pair
Live event: Hormuz closure started ~Feb 28, 2026.
Long equal-weight {VLO, PSX, MPC} vs short SPY from Mar 1, 2026 through present.
Single event -- N=1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AM-1"

def main():
    try:
        tickers = ["VLO", "PSX", "MPC", "SPY"]
        px = load_prices(tickers, start="2026-01-01")
        ret = daily_returns(px)

        needed = ["VLO", "PSX", "MPC", "SPY"]
        avail = [t for t in needed if t in ret.columns and ret[t].dropna().shape[0] > 5]

        refiners = [t for t in ["VLO", "PSX", "MPC"] if t in avail]
        if not refiners or "SPY" not in avail:
            mark_failed(SIGNAL_ID, f"Missing tickers. Available: {avail}",
                        extra={"rule": "Long VLO/PSX/MPC vs short SPY from Mar 2026 (Hormuz closure)",
                               "mechanism": "Strait of Hormuz closure disrupts crude supply, benefiting US refiners with domestic crude access",
                               "source": "AM-1 signal catalog"})
            print(f"FAILED: need refiners + SPY, have {avail}")
            return

        start_date = pd.Timestamp("2026-03-02")
        mask = ret.index >= start_date
        ret_window = ret.loc[mask]

        if ret_window.empty:
            mark_failed(SIGNAL_ID, "No data from Mar 2026 onward",
                        extra={"rule": "Long refiners vs short SPY from Mar 2026",
                               "mechanism": "Hormuz closure benefits US refiners",
                               "source": "AM-1 signal catalog"})
            print("FAILED: no data from Mar 2026")
            return

        # Long refiner basket, short SPY
        long_ret = ret_window[refiners].mean(axis=1)
        short_ret = ret_window["SPY"]
        pair_pnl = (long_ret - short_ret).dropna()

        n_days = len(pair_pnl)
        print(f"Trade period: {pair_pnl.index[0].date()} to {pair_pnl.index[-1].date()} ({n_days} days)")
        print(f"Refiners used: {refiners}")
        print(f"WARNING: N=1 event. Results are descriptive, not predictive.")

        if n_days < 30:
            eq = (1 + pair_pnl).cumprod()
            total_ret = eq.iloc[-1] - 1 if len(eq) > 0 else 0
            ann_est = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
            sharpe_est = float(pair_pnl.mean() / pair_pnl.std() * np.sqrt(252)) if pair_pnl.std() > 0 else 0
            metrics = {
                "name": "AM-1 Hormuz Refiner Pair",
                "n_days": n_days,
                "total_return": float(total_ret),
                "cagr": float(ann_est),
                "sharpe": sharpe_est,
                "start": str(pair_pnl.index[0].date()),
                "end": str(pair_pnl.index[-1].date()),
                "note": f"Only {n_days} days -- annualized figures are extrapolations"
            }
            print(f"  Total pair return: {total_ret*100:.2f}%")
            print(f"  Annualized estimate: {ann_est*100:.2f}%")
        else:
            metrics = compute_metrics(pair_pnl, benchmark=None, name="AM-1 Hormuz Refiner Pair")
            print_metrics(metrics)

        # Also report individual legs
        long_eq = (1 + long_ret.dropna()).cumprod()
        short_eq = (1 + short_ret.dropna()).cumprod()
        if len(long_eq) > 0:
            metrics["refiner_total_return"] = float(long_eq.iloc[-1] - 1)
        if len(short_eq) > 0:
            metrics["spy_total_return"] = float(short_eq.iloc[-1] - 1)

        metrics["refiners_used"] = refiners
        metrics["status"] = "ok"
        metrics["caveat"] = "N=1 geopolitical event. Results describe this specific Hormuz closure episode only."

        save_result(SIGNAL_ID, metrics, extra={
            "rule": f"Long equal-weight {{{', '.join(refiners)}}} vs short SPY from Mar 1, 2026 (Hormuz closure)",
            "mechanism": "Strait of Hormuz closure (~Feb 28, 2026) disrupts ~20% of global crude transit. US refiners (VLO, PSX, MPC) with access to domestic WTI crude benefit from widening crack spreads as global supply tightens. Short SPY hedges broad market risk.",
            "source": "AM-1 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long refiners vs short SPY from Mar 2026",
                           "mechanism": "Hormuz closure benefits US refiners",
                           "source": "AM-1 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
