"""
U8 Mongolia-China border closure events at Gants Mod / Ceke / Erenhot
-> long Teck Resources (TECK).

Rule: When the Mongolia-China land border (Gants Mod / Ceke / Erenhot crossings)
is closed or capacity-curtailed by Chinese authorities, long Teck (TECK)
for 30 trading days (~6 weeks). Non-overlapping.

Mechanism: Mongolia exports ~40 Mt coking coal annually, > 90% via these
crossings to China. Closures spike China seaborne coking-coal demand from
Canada / Australia; Teck (Elk Valley HCC) is the largest non-China benchmark
coking-coal name on US listing.

Source: Mongolian Customs General Administration & Chinese MOFCOM press
releases; press archives (Reuters, S&P Platts) for 2018-2024 closures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated Mongolia-China border closures with material truck-flow impact.
EVENTS = [
    ("2020-02-10", "COVID border closure - first wave"),
    ("2020-11-09", "Gants Mod closed - 2nd wave"),
    ("2021-01-25", "Chinese New Year + COVID closure"),
    ("2021-10-13", "Ulaanbaatar COVID surge - export halt"),
    ("2022-01-31", "Lunar New Year closures + Omicron"),
    ("2022-04-25", "Erenhot 14-day closure (zero-COVID)"),
    ("2023-01-22", "Chinese New Year reduced flows post-reopening"),
    ("2024-02-10", "Lunar NY reduced capacity"),
]


def main():
    try:
        teck = load_prices(["TECK"], start="2015-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U8_mongolia_coking_coal_trucks", f"TECK load failed: {e}")

    if len(teck) < 200:
        return mark_failed("U8_mongolia_coking_coal_trucks", "insufficient TECK history")

    rets = teck.pct_change()

    HOLD = 30
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    used = []
    for d_str, lbl in EVENTS:
        d = pd.Timestamp(d_str)
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U8_mongolia_coking_coal_trucks", "no events landed")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U8 Mongolia border close -> long TECK {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U8_mongolia_coking_coal_trucks", m, extra={
        "status": "ok",
        "rule": "When the Mongolia-China land border (Gants Mod / Ceke / Erenhot) is closed or capacity-curtailed, long TECK for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "Mongolia exports ~40 Mt of coking coal/yr, > 90% via these crossings to China. Closures lift seaborne coking-coal demand from Canada / Australia. Teck's Elk Valley HCC is the most direct US-listed beneficiary.",
        "source": "Curated 2020-2024 closures from press archives (Reuters, S&P Platts) and Mongolian Customs GA. TECK via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N moderate (~8); concentrated in COVID era 2020-2022.",
    })


if __name__ == "__main__":
    main()
