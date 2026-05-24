"""
U4 Norway sea-lice region-3-to-5 forced-harvest events -> long MOWI.OL / short BAKKA.OL.

Rule: When the Mattilsynet sea-lice surveillance program flags a forced-
harvest or biomass-reduction action in Norwegian PO3-PO5 (Vestland) regions,
long MOWI.OL and short BAKKA.OL (Faroese exposure) for 30 trading days
(~6 weeks). Non-overlapping.

Mechanism: Norway accounts for ~50% of global Atlantic salmon supply.
Forced-harvest decisions tighten supply within months and lift spot salmon
prices. Mowi is the largest diversified producer with material West-Norway
exposure; Bakkafrost is Faroe-only so benefits less from Norway-specific
disruption (and underperforms on broad sentiment).

Source: We attempt to scrape barentswatch.no / lakselus.no. If inaccessible
historically, fall back to curated forced-harvest events 2018-2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated PO3-5 forced-harvest / capacity-reduction Mattilsynet decisions.
# Sources: press archives (intrafish.no, salmonbusiness.com).
EVENTS = [
    ("2018-09-15", "Mattilsynet first capacity-reduction rounds PO3-5"),
    ("2020-02-12", "Forced harvest PO4 (Hordaland)"),
    ("2021-09-01", "Trafikklys red zones PO3 / PO4 set"),
    ("2022-09-01", "Trafikklys red zones renewed PO3 / PO4"),
    ("2023-02-15", "PO4 emergency harvest orders"),
    ("2024-09-01", "Trafikklys 2024 - red PO3/PO4"),
]


def main():
    try:
        mowi = load_prices(["MOWI.OL"], start="2016-01-01").iloc[:, 0].dropna()
        bakka = load_prices(["BAKKA.OL"], start="2016-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U4_norway_salmon_lice", f"yfinance load failed: {e}")

    if len(mowi) < 200 or len(bakka) < 200:
        return mark_failed("U4_norway_salmon_lice", "insufficient MOWI/BAKKA history")

    px = pd.concat({"MOWI": mowi, "BAKKA": bakka}, axis=1).sort_index()
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
        pos.iloc[idx:end_idx, pos.columns.get_loc("MOWI")] = 1.0
        pos.iloc[idx:end_idx, pos.columns.get_loc("BAKKA")] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U4_norway_salmon_lice", "no events landed in price history")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets["MOWI"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U4 Mattilsynet PO3-5 -> long MOWI / short BAKKA {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U4_norway_salmon_lice", m, extra={
        "status": "ok",
        "rule": "When Mattilsynet flags forced-harvest / capacity-reduction in Norwegian PO3-PO5, go long MOWI.OL and short BAKKA.OL for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "Norway = ~50% of global Atlantic-salmon supply; forced harvests tighten supply and lift spot salmon. MOWI has West-Norway exposure; BAKKA is Faroe-only and is the relative laggard on Norway-specific stress.",
        "source": "Curated PO3-5 Mattilsynet / Trafikklys decisions 2018-2024 from intrafish.no / salmonbusiness.com (Mattilsynet API at barentswatch.no requires auth; not used).",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~6); pair has high beta to broad salmon-equity tape.",
    })


if __name__ == "__main__":
    main()
