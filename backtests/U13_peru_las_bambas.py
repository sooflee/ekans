"""
U13 Peru Las Bambas community blockade events -> long MMG (1208.HK) + HG=F.

Rule: When a major community-blockade event begins at MMG's Las Bambas copper
mine (Apurimac, Peru) - i.e. one that halts operations or the export
corridor for > 7 days - long MMG (1208.HK) and HG=F (COMEX copper) equally
weighted for 30 trading days (~6 weeks). Non-overlapping.

Mechanism: Las Bambas is ~2% of global mined copper; sustained shut-ins
of mine + corridor squeeze concentrate flow into seaborne; copper futures
rally on the supply scare, and MMG itself re-rates on the resolution-
optionality bid (anchored to mean-revert).

Source: Curated from Reuters / La Republica / MMG ASX / HKEX disclosures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated major Las Bambas blockade-start dates (operations or corridor halted > 7 days).
EVENTS = [
    ("2021-11-20", "Chumbivilcas blockade halts 24-day corridor (Q4 2021)"),
    ("2022-04-20", "51-day mine stoppage begins (Fuerabamba/Huancuire)"),
    ("2022-08-03", "Renewed corridor blockade post-restart"),
    ("2022-12-12", "Pedro Castillo aftermath - corridor blockade"),
    ("2023-01-19", "Apurimac protests escalate - extended 18-day shut"),
    ("2024-03-15", "Local community road block (~10 days)"),
]


def main():
    try:
        mmg = load_prices(["1208.HK"], start="2014-01-01").iloc[:, 0].dropna()
        hg = load_prices(["HG=F"], start="2015-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U13_peru_las_bambas", f"yfinance load failed: {e}")

    if len(mmg) < 200 or len(hg) < 200:
        return mark_failed("U13_peru_las_bambas", "insufficient MMG/HG history")

    px = pd.concat({"MMG": mmg, "HG": hg}, axis=1).sort_index()
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
        return mark_failed("U13_peru_las_bambas", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets["HG"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U13 Las Bambas blockade -> long MMG+HG {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U13_peru_las_bambas", m, extra={
        "status": "ok",
        "rule": "When a Las Bambas community-blockade halts mine operations or the export corridor for > 7 days, long MMG (1208.HK) + HG=F 50/50 for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "Las Bambas = ~2% of global mined copper. Sustained shut-ins squeeze concentrate flow into seaborne; copper futures rally on the supply scare; MMG re-rates on resolution optionality.",
        "source": "Curated from Reuters / La Republica / MMG HKEX disclosures. Prices via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~6); clustered 2021-2024.",
    })


if __name__ == "__main__":
    main()
