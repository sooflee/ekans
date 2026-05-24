"""
U15 China MOFCOM antimony export-license throughput < 30% baseline
    -> long PPTA + UAMY.

Rule: When China MOFCOM monthly antimony export-license throughput is < 30%
of the 2023 monthly baseline (~150 t Sb-equivalent), long Perpetua (PPTA)
and US Antimony (UAMY) equally weighted for 30 trading days (~6 weeks).
Roll monthly on a fresh sub-30% print; non-overlapping windows.

Mechanism: China produces ~50% of mined antimony + > 80% of refined metal.
MOFCOM's Sept 15 2024 license regime created a backlog; sub-30% throughput
months signal sustained export restriction, which directly tightens
non-Chinese availability and lifts Western antimony equity NAV.

Source: MOFCOM customs releases (curated months); PPTA + UAMY via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated months where China antimony exports printed < 30% of 2023 baseline.
# Each month's anchor = first trading day of the SUBSEQUENT month (data publishes lagged).
EVENTS = [
    ("2024-10-21", "Sept 2024 customs - 0 t Sb exports"),
    ("2024-11-20", "Oct 2024 customs - 0 t"),
    ("2024-12-20", "Nov 2024 customs - de minimis"),
    ("2025-01-20", "Dec 2024 customs - sub-30%"),
    ("2025-02-20", "Jan 2025 customs - sub-30%"),
    ("2025-04-21", "Mar 2025 customs - export ban codified"),
    ("2025-06-20", "May 2025 customs - quota minimal"),
    ("2025-09-20", "Aug 2025 customs - sub-30%"),
]


def main():
    try:
        ppta = load_prices(["PPTA"], start="2018-01-01").iloc[:, 0].dropna()
        uamy = load_prices(["UAMY"], start="2015-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U15_china_antimony_licenses", f"yfinance load failed: {e}")

    if len(ppta) < 100 or len(uamy) < 200:
        return mark_failed("U15_china_antimony_licenses", "insufficient PPTA/UAMY history")

    px = pd.concat({"PPTA": ppta, "UAMY": uamy}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 30
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
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
        pos.iloc[idx:end_idx, :] = 0.5
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U15_china_antimony_licenses", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U15 China Sb licence backlog -> long PPTA+UAMY {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U15_china_antimony_licenses", m, extra={
        "status": "ok",
        "rule": "When China MOFCOM monthly antimony export throughput is < 30% of 2023 baseline (per customs releases), long an equal-weight basket of PPTA + UAMY for 30 trading days; non-overlapping.",
        "mechanism": "China = ~50% of mined antimony + > 80% refined; MOFCOM's Sept 2024 license regime created a sustained backlog. Sub-30% months directly tighten non-Chinese availability and lift Western antimony NAV.",
        "source": "Curated MOFCOM customs months; PPTA + UAMY via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~6-8); single-regime cluster Oct 2024-2025.",
    })


if __name__ == "__main__":
    main()
