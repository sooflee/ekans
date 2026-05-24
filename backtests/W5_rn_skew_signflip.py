"""
W5 Risk-Neutral Skew Sign-Flip (realized-skew simplification).

Original signal (Bali et al. 2023 / paper trail): the cross-section of
risk-neutral skew (extracted from option chains) flips sign at the daily
horizon. We don't have daily option chains for thousands of stocks, so we
implement a SIMPLIFIED realized-skewness version:

  - For each S&P 500 stock (subset), compute the realized skewness of daily
    returns over the trailing 21 trading days.
  - Monthly rebalance: rank cross-sectionally, long the TOP decile of
    realized skewness (positive skew names), short the BOTTOM decile
    (negative skew names).
  - Equal-weight legs, dollar-neutral.

Honest note: realized skewness ≠ risk-neutral skewness, and the published
sign-flip result is a daily option-implied phenomenon. Here we test only
whether a monthly cross-sectional realized-skewness sort has a payoff in
the recent S&P universe. This is a documentation-only proxy.

Universe: 30 large S&P 500 names that we already have cached.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "NFLX", "XOM", "CVX",
    "PFE", "KO", "PEP", "WMT", "T", "VZ", "INTC", "CSCO", "CRM", "ABT",
    "ORCL", "WFC", "TMO", "ABBV", "MRK", "COST", "MCD", "NKE", "QCOM", "IBM",
]


def main():
    try:
        px = load_prices(UNIVERSE + ["SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed("W5_rn_skew_signflip", f"Price load failed: {e}")

    rets = px.pct_change().dropna(how="all")
    if rets.empty or rets.shape[0] < 252:
        return mark_failed("W5_rn_skew_signflip",
                            f"Returns matrix too short: {rets.shape}")

    universe = [c for c in UNIVERSE if c in rets.columns]
    if len(universe) < 20:
        return mark_failed("W5_rn_skew_signflip",
                            f"Only {len(universe)} tickers loaded")
    R = rets[universe]
    spy_ret = rets["SPY"]

    # Realized skewness on a 21-trading-day rolling window
    win = 21
    rs = R.rolling(win).skew()

    # Monthly rebalance: take month-end positions
    me_idx = R.resample("ME").last().index
    me_skew = rs.reindex(me_idx).dropna(how="all")
    me_skew = me_skew.dropna(thresh=int(0.7 * len(universe)))

    if len(me_skew) < 12:
        return mark_failed("W5_rn_skew_signflip",
                            f"Only {len(me_skew)} monthly rebalances")

    # Decile portfolios: top decile long, bottom decile short
    positions = pd.DataFrame(0.0, index=me_skew.index, columns=universe)
    for d in me_skew.index:
        row = me_skew.loc[d].dropna()
        if len(row) < 10:
            continue
        n_lo = max(1, len(row) // 10)
        n_hi = n_lo
        top = row.nlargest(n_hi).index
        bot = row.nsmallest(n_lo).index
        positions.loc[d, top] = 1.0 / n_hi
        positions.loc[d, bot] = -1.0 / n_lo

    # Forward-fill positions to daily; apply at next trading day return
    daily_pos = positions.reindex(R.index, method="ffill").fillna(0)
    pnl = (daily_pos.shift(1) * R).sum(axis=1).dropna()
    pnl = pnl[pnl.index >= me_skew.index[0]]

    if len(pnl) < 100:
        return mark_failed("W5_rn_skew_signflip",
                            f"Too few PnL days: {len(pnl)}")

    bench = spy_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name="W5 realized-skew decile L/S (simplification)")
    m["universe_size"] = int(len(universe))
    m["rebalances"] = int(len(me_skew))
    print_metrics(m)

    save_result("W5_rn_skew_signflip", m, extra={
        "status": "ok_simplification",
        "rule": ("Monthly: rank S&P 500 names by trailing-21d realized skewness. "
                  "Long top decile, short bottom decile (equal-weight, dollar-neutral)."),
        "mechanism": ("The published RN-skew sign-flip is option-implied: short-"
                       "horizon RN-skew predicts returns with one sign and "
                       "long-horizon predicts the opposite. We can only test "
                       "realized skewness as a free proxy. The realized-skew "
                       "long-top-short-bottom convention here follows Bali-Hu-"
                       "Murray 2019 'Cross-section of stock returns and "
                       "skewness'."),
        "source": ("Spec calls for daily option chains across 1000s of stocks "
                    "(not free). This implementation uses realized skewness on "
                    "40 large S&P names instead — a known weaker but documentable "
                    "proxy."),
        "universe": ", ".join(universe),
        "caveats": ("Realized skewness ≠ risk-neutral skewness. The 'sign flip' "
                    "across horizons is specifically an option-implied result; "
                    "this test only addresses the monthly realized-skew cross-section."),
    })


if __name__ == "__main__":
    main()
