"""
AL-2  Immigration -> Homebuilder vs HD/LOW
Regime trade: short XHB vs long (HD + LOW)/2 from Jan 2025 through present.
Tests whether immigration policy tightening hurts homebuilders more than home improvement.
Short timeframe (~5 months).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AL-2"

def main():
    try:
        tickers = ["XHB", "HD", "LOW", "SPY"]
        px = load_prices(tickers, start="2024-09-01")
        ret = daily_returns(px)

        needed = ["XHB", "HD", "LOW"]
        avail = [t for t in needed if t in ret.columns and ret[t].dropna().shape[0] > 10]
        if len(avail) < 3:
            mark_failed(SIGNAL_ID, f"Missing tickers: need XHB, HD, LOW; got {avail}",
                        extra={"rule": "Short XHB vs Long HD/LOW from Jan 2025",
                               "mechanism": "Immigration restriction reduces construction labor, hurting homebuilders; HD/LOW benefit from existing homeowners doing DIY",
                               "source": "AL-2 signal catalog"})
            print(f"FAILED: only have {avail}")
            return

        start_date = pd.Timestamp("2025-01-02")
        mask = ret.index >= start_date
        ret_window = ret.loc[mask]

        # Long leg: equal-weight HD + LOW
        long_ret = ret_window[["HD", "LOW"]].mean(axis=1)
        # Short leg: XHB
        short_ret = ret_window["XHB"]

        # Pair PnL: long (HD+LOW)/2 - short XHB
        pair_pnl = (long_ret - short_ret).dropna()
        spy_ret = ret_window["SPY"].reindex(pair_pnl.index) if "SPY" in ret_window.columns else None

        n_days = len(pair_pnl)
        print(f"Trade period: {n_days} days ({pair_pnl.index[0].date()} to {pair_pnl.index[-1].date()})")
        print(f"NOTE: Very short timeframe (~{n_days} trading days). Results are anecdotal, not statistically robust.")

        if n_days < 30:
            # compute_metrics will flag insufficient data
            eq = (1 + pair_pnl).cumprod()
            total_ret = eq.iloc[-1] - 1 if len(eq) > 0 else 0
            metrics = {
                "name": "AL-2 Immigration Homebuilder Pair",
                "n_days": n_days,
                "total_return": float(total_ret),
                "note": f"Only {n_days} trading days -- too short for reliable metrics. Total return shown.",
                "start": str(pair_pnl.index[0].date()),
                "end": str(pair_pnl.index[-1].date()),
            }
            # Annualize rough estimate
            if n_days > 0:
                ann = (1 + total_ret) ** (252 / n_days) - 1
                metrics["cagr_annualized_estimate"] = float(ann)
                metrics["cagr"] = float(ann)
                metrics["sharpe"] = float(pair_pnl.mean() / pair_pnl.std() * np.sqrt(252)) if pair_pnl.std() > 0 else 0
            print(f"  Total return: {total_ret*100:.2f}%")
            print(f"  Annualized estimate: {ann*100:.2f}%")
        else:
            metrics = compute_metrics(pair_pnl, benchmark=spy_ret, name="AL-2 Immigration Homebuilder Pair")
            print_metrics(metrics)

        metrics["status"] = "ok"
        metrics["tickers_used"] = needed
        metrics["caveat"] = f"Very short backtest ({n_days} trading days). Treat as anecdotal N=1 regime observation."

        save_result(SIGNAL_ID, metrics, extra={
            "rule": "Short XHB vs Long (HD+LOW)/2 from Jan 2025 (immigration policy tightening)",
            "mechanism": "Immigration restriction reduces construction labor supply, disproportionately hurting homebuilders (XHB) vs home improvement retailers (HD, LOW) which benefit from existing homeowners investing in their current homes instead of moving.",
            "source": "AL-2 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Short XHB vs Long HD/LOW from Jan 2025",
                           "mechanism": "Immigration restriction hurts homebuilders",
                           "source": "AL-2 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
