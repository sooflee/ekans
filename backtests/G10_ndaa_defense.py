"""
G-10 NDAA enactment defense drift.

Hardcoded NDAA signature (presidential signing) dates since 2010, sourced from Congress.gov /
NDAA Wikipedia entries.
For each year: long ITA from D-10 trading days through D+3 trading days, where D = NDAA signing date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result

# NDAA Public Law signature dates (President's signing). Source: Congress.gov / Wikipedia.
# Format: fiscal-year (FY) -> signing date.
NDAA_SIGN = {
    "FY2010": "2009-10-28",
    "FY2011": "2011-01-07",
    "FY2012": "2011-12-31",
    "FY2013": "2013-01-02",
    "FY2014": "2013-12-26",
    "FY2015": "2014-12-19",
    "FY2016": "2015-11-25",
    "FY2017": "2016-12-23",
    "FY2018": "2017-12-12",
    "FY2019": "2018-08-13",
    "FY2020": "2019-12-20",
    "FY2021": "2021-01-01",  # override veto, considered law Jan 1
    "FY2022": "2021-12-27",
    "FY2023": "2022-12-23",
    "FY2024": "2023-12-22",
    "FY2025": "2024-12-23",
}


def main():
    px = load_prices(["ITA"], start="2008-01-01")["ITA"]
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    held = 0
    for fy, d in NDAA_SIGN.items():
        d = pd.Timestamp(d)
        loc = idx.searchsorted(d)
        # Map to nearest trading day
        if loc >= len(idx):
            continue
        D = idx[loc]
        i = idx.get_loc(D)
        start = max(0, i - 10)
        end = min(len(idx) - 1, i + 3)
        pos.iloc[start:end + 1] = 1.0
        held += 1

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    spy = load_prices(["SPY"], start="2008-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G10 NDAA defense drift")
    print_metrics(m)
    save_result("G10_ndaa_defense", m, extra={
        "status": "ok",
        "rule": "Long ITA from D-10 to D+3 (trading days), D = NDAA signing date.",
        "universe": "ITA",
        "n_events": len(NDAA_SIGN),
        "n_events_held": held,
        "pct_days_long": float(pos.mean()),
        "source": "NDAA Wikipedia / Congress.gov signing dates",
        "notes": "Small N (16 events).",
    })


if __name__ == "__main__":
    main()
