"""
G-18 PBOC RRR (Reserve Requirement Ratio) cut -> Hang Seng/China momentum.

Hardcoded major RRR cut announcements 2015-2024 (selected from PBOC announcements). For each: long
FXI at next session open, exit day 10.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result

# Announcement dates (effective date is usually 1-2 weeks later). Sources: PBOC press releases / WSJ /
# Reuters coverage. We use the announcement date.
RRR_CUTS = [
    "2015-02-04",
    "2015-04-19",
    "2015-06-27",
    "2015-08-25",
    "2015-10-23",
    "2016-02-29",
    "2018-04-17",
    "2018-06-24",
    "2018-10-07",
    "2019-01-04",
    "2019-09-06",
    "2020-01-01",
    "2020-03-13",
    "2020-04-03",
    "2021-07-09",
    "2021-12-06",
    "2022-04-15",
    "2022-11-25",
    "2023-03-17",
    "2023-09-14",
    "2024-01-24",
    "2024-09-24",
]


def main():
    px = load_prices(["FXI"], start="2014-01-01")["FXI"]
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    used = []
    for d in RRR_CUTS:
        D = pd.Timestamp(d)
        loc = idx.searchsorted(D, side="right")
        if loc >= len(idx):
            continue
        start = loc       # next session
        end = min(loc + 10, len(idx) - 1)
        pos.iloc[start:end + 1] = 1.0
        used.append(d)

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    spy = load_prices(["SPY"], start="2014-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G18 PBOC RRR -> FXI")
    print_metrics(m)
    save_result("G18_pboc_rrr_fxi", m, extra={
        "status": "ok",
        "rule": "On each PBOC RRR cut announcement date, long FXI at next session open, exit after "
                "10 trading days.",
        "universe": "FXI",
        "n_events": len(RRR_CUTS),
        "events": used,
        "pct_days_long": float(pos.mean()),
        "source": "PBOC press releases (curated)",
    })


if __name__ == "__main__":
    main()
