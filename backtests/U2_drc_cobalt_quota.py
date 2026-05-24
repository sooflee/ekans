"""
U2 DRC ARECOMS cobalt export-quota / export-ban events -> long cobalt complex.

Rule: When DRC ARECOMS announces a cobalt export ban or quota, long CMOC
(China Molybdenum, '603993.SS') equally weighted with LIT ETF (poor but
liquid cobalt-exposed proxy) for 40 trading days (~8 weeks). Non-overlapping.

Mechanism: DRC supplies ~75% of mined cobalt; export curtailments tighten
the cobalt supply chain, lift cobalt metal & hydroxide prices, and re-rate
cobalt-exposed equities. CMOC's TFM / KFM mines in DRC sit on stockpile that
must wait for quota allocations - paradoxically benefits incumbents holding
existing inventory and remaining quota.

Source: ARECOMS press releases, DRC presidential decrees (Feb 22 2025 ban
extended in June 2025 and quota system Oct 2025). Curated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated ARECOMS / DRC presidential cobalt-supply events.
EVENTS = [
    ("2025-02-22", "ARECOMS announces 4-month cobalt export ban"),
    ("2025-06-21", "Ban extended for additional 3 months"),
    ("2025-09-22", "Ban replaced by 18,125 t annual export quota regime"),
    ("2025-10-16", "Quota allocation rules published - quota cuts vs 2024 baseline"),
]


def main():
    cmoc = None
    lit = None
    try:
        cmoc_raw = load_prices(["603993.SS"], start="2018-01-01")
        cmoc = cmoc_raw.iloc[:, 0].dropna()
    except Exception:
        pass
    try:
        lit_raw = load_prices(["LIT"], start="2018-01-01")
        lit = lit_raw.iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U2_drc_cobalt_quota", f"LIT load failed: {e}")

    # Build returns frame
    cols = {}
    if cmoc is not None and len(cmoc) > 100:
        cols["CMOC"] = cmoc
    if lit is not None and len(lit) > 100:
        cols["LIT"] = lit
    if not cols:
        return mark_failed("U2_drc_cobalt_quota", "no usable price series")

    px = pd.concat(cols, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 40  # ~8 weeks
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    n_events = 0
    last_end = None
    used = []
    weight = 1.0 / len(rets.columns)
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
        pos.iloc[idx:end_idx, :] = weight
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U2_drc_cobalt_quota", "no events landed in price history")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U2 DRC cobalt curb -> long CMOC+LIT {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}; tickers: {list(rets.columns)}")
    print_metrics(m)

    save_result("U2_drc_cobalt_quota", m, extra={
        "status": "ok",
        "rule": "When DRC ARECOMS / presidency announces a cobalt export ban, ban extension, or quota allocation event, long an equal-weight basket of CMOC (603993.SS) + LIT for 40 trading days (~8 weeks); non-overlapping.",
        "mechanism": "DRC supplies ~75% of mined cobalt. Export curtailments tighten the global cobalt supply chain, re-rate cobalt prices, and benefit incumbents holding stockpile / quota allocations.",
        "source": "Curated ARECOMS press releases & DRC presidential decrees (Feb / Jun / Sep / Oct 2025). CMOC and LIT via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N very small (~3-4); 2025-only event cluster, results are illustrative only.",
    })


if __name__ == "__main__":
    main()
