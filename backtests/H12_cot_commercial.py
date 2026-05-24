"""
H12 COT commercial-hedger positioning extreme.

CFTC Disaggregated COT (resource 6dca-aqww via publicreporting.cftc.gov).
For each underlying we compute commercial NET position
  net = comm_positions_long_all - comm_positions_short_all
then rolling 156-week (3-year) percentile of `net`.

Rule (per underlying):
  - net > 90th percentile -> long the underlying ETF for 8 weeks
  - net < 10th percentile -> short the underlying ETF for 8 weeks
  - else flat

Underlyings:
  - GOLD    -> trade GLD
  - E-MINI S&P 500 -> trade SPY
Combined portfolio is the equal-weighted sum of the two leg PnLs.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


def fetch_cot(contract_name):
    safe = contract_name.replace(" ", "_").replace("&", "and")
    cache = DATA / f"cftc_disagg_{safe}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get(
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json",
        params={
            "contract_market_name": contract_name,
            "$limit": 50000,
            "$order": "report_date_as_yyyy_mm_dd ASC",
        },
        timeout=120,
    )
    j = r.json()
    df = pd.DataFrame(j)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["comm_long"] = pd.to_numeric(df["comm_positions_long_all"], errors="coerce")
    df["comm_short"] = pd.to_numeric(df["comm_positions_short_all"], errors="coerce")
    df["net"] = df["comm_long"] - df["comm_short"]
    df = df.set_index("date")[["comm_long", "comm_short", "net"]].sort_index()
    df.to_parquet(cache)
    return df


def signal_for(contract_name, etf_ticker, start="2005-01-01"):
    cot = fetch_cot(contract_name)
    if cot.empty:
        return None, etf_ticker
    cot = cot.loc[start:]
    net = cot["net"].dropna()

    # Need a non-trivial history. The "rolling 3-year percentile":
    # for each week d, percentile of net.loc[d] within net.loc[d-156w : d].
    win = 156  # weeks
    def pct(s):
        # percentile rank of last value (0..1) in window s
        if len(s) < 30:
            return np.nan
        return (s.rank(pct=True)).iloc[-1]
    pct_rank = net.rolling(window=win).apply(pct, raw=False)

    # signal at each weekly report
    sig = pd.Series(0, index=net.index)
    sig[pct_rank > 0.90] = 1
    sig[pct_rank < 0.10] = -1

    # Convert to daily position with 8-week hold.
    # If a new signal arrives during a hold, the new signal overrides
    # and resets the 8-week clock.
    weekly_idx = net.index
    state = 0
    weeks_left = 0
    weekly_pos = pd.Series(0.0, index=weekly_idx)
    for d in weekly_idx:
        s = sig.loc[d]
        if s != 0:
            state = s
            weeks_left = 8
        else:
            weeks_left -= 1
            if weeks_left <= 0:
                state = 0
                weeks_left = 0
        weekly_pos.loc[d] = state

    px = load_prices([etf_ticker], start=start).iloc[:, 0]
    rets = px.pct_change()
    pos_d = weekly_pos.reindex(px.index, method="ffill").fillna(0.0)
    pnl = (pos_d.shift(1) * rets).dropna()
    return pnl, etf_ticker


def main():
    legs = {}
    for cn, etf in [("GOLD", "GLD"), ("E-MINI S&P 500", "SPY")]:
        try:
            pnl, t = signal_for(cn, etf)
            if pnl is not None and not pnl.empty:
                legs[etf] = pnl
        except Exception as e:
            print("leg err", cn, e)

    if not legs:
        return mark_failed("H12_cot_commercial", "no CFTC legs computed")

    # combine 50/50
    combined = pd.concat(legs, axis=1).fillna(0.0).mean(axis=1)
    bench = load_prices(["SPY"], start="2005-01-01").iloc[:, 0].pct_change()
    m = compute_metrics(combined.dropna(), benchmark=bench.loc[combined.index],
                        name="H12 COT commercial-hedger extremes")
    print_metrics(m)
    save_result("H12_cot_commercial", m, extra={
        "status": "ok",
        "rule": "Per underlying: commercial NET position pct-rank>0.9 -> long ETF 8w; <0.1 -> short ETF 8w. Portfolio = equal-wt of GLD and SPY legs.",
        "data_source": "CFTC Disaggregated COT via publicreporting.cftc.gov resource 6dca-aqww.json.",
        "legs": list(legs.keys()),
    })


if __name__ == "__main__":
    main()
