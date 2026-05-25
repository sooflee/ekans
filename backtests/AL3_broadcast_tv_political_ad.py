"""
AL-3  Broadcast TV Political Ad Pricing -> Long NXST/GTN (seasonal May-Nov election years)
In each federal election year, long equal-weight {NXST, GTN} from May 1 through Nov 15.
Election years: 2018, 2020, 2022, 2024.  Off-cycle years are NOT traded.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AL-3"
ELECTION_YEARS = [2018, 2020, 2022, 2024]
ENTRY_MMDD = "05-01"
EXIT_MMDD = "11-15"

def main():
    try:
        # Try to load both tickers
        tickers_long = ["NXST", "GTN"]
        px = load_prices(tickers_long + ["SPY"], start="2017-01-01")
        ret = daily_returns(px)

        available = [t for t in tickers_long if t in ret.columns and ret[t].dropna().shape[0] > 50]
        if not available:
            mark_failed(SIGNAL_ID, "No price data for NXST or GTN",
                        extra={"rule": "Long NXST/GTN May-Nov election years",
                               "mechanism": "Political ad spend boosts local TV station revenue in election cycles",
                               "source": "AL-3 signal catalog"})
            print("FAILED: no data for NXST or GTN")
            return

        print(f"Available tickers: {available}")

        all_pnl = []
        season_results = []

        for yr in ELECTION_YEARS:
            entry = pd.Timestamp(f"{yr}-{ENTRY_MMDD}")
            exit_ = pd.Timestamp(f"{yr}-{EXIT_MMDD}")
            mask = (ret.index >= entry) & (ret.index <= exit_)
            chunk = ret.loc[mask]
            if chunk.empty:
                print(f"  {yr}: no data in window")
                continue

            # equal-weight long basket
            basket = chunk[available].mean(axis=1)
            spy_chunk = chunk["SPY"] if "SPY" in chunk.columns else None

            eq = (1 + basket).cumprod()
            season_ret = eq.iloc[-1] - 1 if len(eq) > 0 else 0.0

            spy_eq = (1 + spy_chunk).cumprod() if spy_chunk is not None and len(spy_chunk) > 0 else None
            spy_ret = spy_eq.iloc[-1] - 1 if spy_eq is not None and len(spy_eq) > 0 else None

            season_results.append({
                "year": yr,
                "basket_return": float(season_ret),
                "spy_return": float(spy_ret) if spy_ret is not None else None,
                "excess": float(season_ret - spy_ret) if spy_ret is not None else None,
                "n_days": len(chunk)
            })
            print(f"  {yr}: basket={season_ret*100:.1f}%  SPY={spy_ret*100:.1f}%  excess={((season_ret-spy_ret)*100):.1f}%  ({len(chunk)} days)")

            all_pnl.append(basket)

        if not all_pnl:
            mark_failed(SIGNAL_ID, "No election-year data available",
                        extra={"rule": "Long NXST/GTN May-Nov election years",
                               "mechanism": "Political ad spend boosts local TV station revenue in election cycles",
                               "source": "AL-3 signal catalog"})
            print("FAILED: no election-year data")
            return

        # Concatenate all seasonal PnL strips
        combined_pnl = pd.concat(all_pnl)
        combined_spy = pd.concat([ret.loc[(ret.index >= pd.Timestamp(f"{yr}-{ENTRY_MMDD}")) &
                                          (ret.index <= pd.Timestamp(f"{yr}-{EXIT_MMDD}")), "SPY"]
                                  for yr in ELECTION_YEARS if "SPY" in ret.columns])

        metrics = compute_metrics(combined_pnl, benchmark=combined_spy, name="AL-3 TV Ad Election Basket")
        print_metrics(metrics)

        # Compute average seasonal return
        avg_basket = np.mean([s["basket_return"] for s in season_results])
        avg_spy = np.mean([s["spy_return"] for s in season_results if s["spy_return"] is not None])
        avg_excess = np.mean([s["excess"] for s in season_results if s["excess"] is not None])

        metrics["avg_season_return"] = float(avg_basket)
        metrics["avg_spy_season_return"] = float(avg_spy)
        metrics["avg_excess_season_return"] = float(avg_excess)
        metrics["season_details"] = season_results
        metrics["tickers_used"] = available
        metrics["status"] = "ok"

        save_result(SIGNAL_ID, metrics, extra={
            "rule": "Long equal-weight NXST/GTN May 1 - Nov 15 in federal election years (2018,2020,2022,2024)",
            "mechanism": "Political ad spend surges in election years, boosting local TV station revenue. Broadcast TV stations (NXST, GTN) are the primary beneficiaries.",
            "source": "AL-3 signal catalog"
        })
        print(f"\nSaved {SIGNAL_ID}")

    except Exception as e:
        mark_failed(SIGNAL_ID, str(e),
                    extra={"rule": "Long NXST/GTN May-Nov election years",
                           "mechanism": "Political ad spend boosts local TV station revenue",
                           "source": "AL-3 signal catalog"})
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
