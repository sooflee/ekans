"""
U10 Chile copper / mining royalty + glacier bill milestones -> long HG=F + Antofagasta.

Rule: When the Chilean Senate or Chamber of Deputies passes a milestone
(primer informe / aprobacion en sala) of a copper-royalty bill or glacier
bill (which together gate roughly 30% of Chilean copper output's marginal
cost), long HG=F (COMEX copper) and ANTO.L (Antofagasta) equally weighted
for 126 trading days (~6 months). Non-overlapping.

Mechanism: Chile produces ~26% of mined copper globally. Royalty hikes /
glacier protection raise marginal cost of supply over a multi-year horizon;
in the short term, bill milestones often trigger short-cover and forecast
revisions. Antofagasta carries pure Chilean exposure with diversified-major
liquidity.

Source: Camara de Diputados / Senado de Chile boletin tracker (curated).
Copper + ANTO via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated Chilean royalty-bill / glacier-bill milestone dates.
EVENTS = [
    ("2021-05-06", "Chamber approves royalty bill (Boletin 12093-08)"),
    ("2022-07-08", "Boric govt 'New Mining Royalty' bill announced"),
    ("2023-05-17", "Senate approves mining royalty (Boletin 12093-08)"),
    ("2024-01-01", "Royalty law effective"),
    ("2022-08-26", "Glacier bill primer informe Comision MA"),
]


def main():
    try:
        hg = load_prices(["HG=F"], start="2015-01-01").iloc[:, 0].dropna()
        anto = load_prices(["ANTO.L"], start="2010-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U10_chile_copper_royalty_bills", f"yfinance load failed: {e}")

    if len(hg) < 200 or len(anto) < 200:
        return mark_failed("U10_chile_copper_royalty_bills", "insufficient price history")

    px = pd.concat({"HG": hg, "ANTO": anto}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 126
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
        return mark_failed("U10_chile_copper_royalty_bills", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets["HG"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U10 Chile royalty bill milestone -> long HG+ANTO {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U10_chile_copper_royalty_bills", m, extra={
        "status": "ok",
        "rule": "When the Chilean Senate or Chamber of Deputies passes a milestone vote on a copper-royalty or glacier-protection bill, long an equal-weight basket of HG=F + ANTO.L for 126 trading days (~6 months); non-overlapping.",
        "mechanism": "Chile = ~26% of mined copper; royalty/glacier bills raise marginal cost over multi-year horizon and trigger short-term short-cover & forecast revisions. ANTO is the liquid pure-Chile exposure.",
        "source": "Curated milestone dates from Boletin tracker (Senado/Camara). Prices via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N very small (~3-5); bill cycle is unique 2021-2024.",
    })


if __name__ == "__main__":
    main()
