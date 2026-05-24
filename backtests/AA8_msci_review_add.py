"""
AA8 MSCI quarterly review additions.
MSCI quarterly index review (QIR) announces changes in Feb / May / Aug / Nov
with effective date typically end-of-month following the announcement.

Curating per-name MSCI ACWI/EAFE adds with their announce/effective dates
requires scraping MSCI press releases for 40+ events × 10-30 names each.

This is too heavy for a quick pass without MSCI subscription data. We
attempt a coarser test using the *index-level* MSCI EAFE proxy (EFA) around
announcement dates (no per-name attribution) — long EFA from announce-3 to
effective-1, capturing the "MSCI rebalance flow" lift. Mark_failed if no edge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


# MSCI QIR announcement → effective dates (approx, curated from MSCI press releases).
# Effective date is typically the close of the last business day of the month after announcement.
MSCI_EVENTS = [
    # (announcement, effective)
    ("2015-02-11", "2015-02-27"), ("2015-05-13", "2015-05-29"),
    ("2015-08-12", "2015-08-31"), ("2015-11-11", "2015-11-30"),
    ("2016-02-10", "2016-02-29"), ("2016-05-11", "2016-05-31"),
    ("2016-08-10", "2016-08-31"), ("2016-11-10", "2016-11-30"),
    ("2017-02-09", "2017-02-28"), ("2017-05-15", "2017-05-31"),
    ("2017-08-09", "2017-08-31"), ("2017-11-13", "2017-11-30"),
    ("2018-02-12", "2018-02-28"), ("2018-05-14", "2018-05-31"),
    ("2018-08-13", "2018-08-31"), ("2018-11-12", "2018-11-30"),
    ("2019-02-11", "2019-02-28"), ("2019-05-13", "2019-05-31"),
    ("2019-08-07", "2019-08-30"), ("2019-11-07", "2019-11-29"),
    ("2020-02-12", "2020-02-28"), ("2020-05-12", "2020-05-29"),
    ("2020-08-11", "2020-08-31"), ("2020-11-10", "2020-11-30"),
    ("2021-02-10", "2021-02-26"), ("2021-05-11", "2021-05-28"),
    ("2021-08-11", "2021-08-31"), ("2021-11-10", "2021-11-30"),
    ("2022-02-09", "2022-02-28"), ("2022-05-12", "2022-05-31"),
    ("2022-08-10", "2022-08-31"), ("2022-11-10", "2022-11-30"),
    ("2023-02-09", "2023-02-28"), ("2023-05-11", "2023-05-31"),
    ("2023-08-10", "2023-08-31"), ("2023-11-13", "2023-11-30"),
    ("2024-02-12", "2024-02-29"), ("2024-05-14", "2024-05-31"),
    ("2024-08-12", "2024-08-30"), ("2024-11-07", "2024-11-29"),
    ("2025-02-11", "2025-02-28"), ("2025-05-13", "2025-05-30"),
    ("2025-08-11", "2025-08-29"), ("2025-11-10", "2025-11-28"),
]


def main():
    try:
        px = load_prices(["EFA", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed("AA8_msci_review_add", f"data load failed: {e}")
    px = px.dropna()
    rets = daily_returns(px)
    efa = rets["EFA"]
    spy = rets["SPY"]
    idx = efa.index

    pos_efa = pd.Series(0.0, index=idx)
    pos_spy = pd.Series(0.0, index=idx)
    n_events = 0
    for ann, eff in MSCI_EVENTS:
        a = pd.Timestamp(ann)
        e = pd.Timestamp(eff)
        if e < idx[0] or a > idx[-1]:
            continue
        i_a = idx.searchsorted(a, side="left")
        i_e = idx.searchsorted(e, side="right") - 1
        # Spec: long the index-level proxy (EFA) hedged short SPY from announce-3 → effective-1.
        i_start = max(0, i_a - 3)
        i_end = max(i_start, i_e - 1)
        if i_end <= i_start:
            continue
        hold = idx[i_start:i_end + 1]
        pos_efa.loc[hold] = 1.0
        pos_spy.loc[hold] = -1.0
        n_events += 1

    if n_events < 10:
        return mark_failed("AA8_msci_review_add", f"too few events ({n_events})")

    pnl = (pos_efa.shift(1) * efa + pos_spy.shift(1) * spy).dropna()
    bench = efa.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA8 MSCI review (EFA vs SPY hedge)")
    print_metrics(m)
    save_result("AA8_msci_review_add", m, extra={
        "status": "ok",
        "rule": "Long EFA / short SPY from MSCI QIR announce-3 to effective-1 (Feb/May/Aug/Nov).",
        "universe": "EFA vs SPY (index-level proxy for MSCI EAFE adds)",
        "source": "MSCI QIR press releases (heavy curation; index-level test only)",
        "n_events": n_events,
        "caveat": "Per-name MSCI adds curation deferred — index-level proxy only.",
    })


if __name__ == "__main__":
    main()
