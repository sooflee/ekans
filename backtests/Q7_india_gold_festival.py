"""
Q7 India August gold imports -> festival season GLD trade.

Hardcoded India gold imports for August each year (in tonnes), per public press
reports (Ministry of Commerce, World Gold Council).

Rule: when Aug imports > 80 tonnes, enter long GLD on Sep 8 (2nd week of Sep)
for 45 trading days. Non-overlapping (annual cadence).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


# India August gold imports, tonnes. Sources: WGC quarterly reports, Reuters/PTI press,
# Indian Ministry of Commerce DGFT data. Best-effort historical values.
INDIA_AUG_IMPORTS = {
    2015: 102.0,
    2016: 60.0,    # weak monsoon and PMLA
    2017: 73.0,
    2018: 90.0,
    2019: 30.0,    # high duty + rupee weakness
    2020: 64.0,
    2021: 121.0,
    2022: 60.0,    # duty hike (Jul'22)
    2023: 86.0,
    2024: 132.0,
}

THRESHOLD = 80.0


def main():
    try:
        gld = load_prices(["GLD"], start="2005-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("Q7_india_gold_festival", f"GLD load failed: {e}")
    rets = gld.pct_change()

    events = []
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for year, tons in INDIA_AUG_IMPORTS.items():
        if tons <= THRESHOLD:
            continue
        target = pd.Timestamp(year, 9, 8)
        # first trading day on/after target
        nxt = rets.index[rets.index >= target]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        idx = rets.index.get_loc(start)
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(idx + 45, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        events.append((year, tons, start.date()))

    if n_events == 0:
        return mark_failed("Q7_india_gold_festival",
                           "no qualifying Aug-imports>80t years")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q7 India festival gold -> long GLD 45d")
    m["n_events"] = n_events
    print(f"Events: {n_events}")
    for e in events:
        print(f"  {e[0]} Aug imports {e[1]:.0f}t -> long GLD from {e[2]}")
    print_metrics(m)

    save_result("Q7_india_gold_festival", m, extra={
        "status": "ok",
        "rule": "When India's August gold imports > 80 tonnes, long GLD entering Sep 8 (2nd week) for 45 trading days; non-overlapping by year.",
        "mechanism": "August imports reload Indian jewelers' stocks; Sep-Nov is Diwali/Dhanteras peak demand -> physical premium and spot drift.",
        "source": "Hardcoded from press releases of Indian Ministry of Commerce DGFT / WGC quarterly demand notes; GLD via yfinance.",
        "n_events": n_events,
        "events_table": [{"year": e[0], "aug_imports_t": e[1], "entry": str(e[2])} for e in events],
    })


if __name__ == "__main__":
    main()
