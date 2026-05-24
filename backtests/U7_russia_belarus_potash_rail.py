"""
U7 Russia / Belarus potash rail-transit shock -> long NTR + MOS.

Rule: When Lithuanian rail data show Belaruskali potash transit through
Klaipeda port falls > 20% YoY for 2 consecutive months, long Nutrien (NTR)
and Mosaic (MOS) equally weighted for 90 trading days (~4.5 months).
Non-overlapping.

Mechanism: Russia + Belarus together ~38% of mined potash. Belaruskali
historically routed via Klaipeda; Lithuanian sanctions Feb 2022 severed that
corridor, tightening seaborne potash supply, and re-rated NTR / MOS to
all-time highs.

Source: Lithuanian Railways (litrail.lt) monthly statistics if accessible;
otherwise curated from press archives (Reuters / Argus) for sustained
sanction / transit disruptions. NTR + MOS via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated transit-disruption events.
EVENTS = [
    ("2021-12-08", "Lithuania announces Belaruskali contract termination"),
    ("2022-02-01", "Belaruskali transit halt - first reduction month"),
    ("2022-04-15", "Russia formal potash export limits"),
    ("2023-01-15", "OFAC adds Belaruskali (SDN list); structural shut"),
]


def main():
    try:
        ntr = load_prices(["NTR"], start="2018-01-01").iloc[:, 0].dropna()
        mos = load_prices(["MOS"], start="2010-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U7_russia_belarus_potash_rail", f"yfinance load failed: {e}")

    if len(ntr) < 200 or len(mos) < 200:
        return mark_failed("U7_russia_belarus_potash_rail", "insufficient NTR/MOS history")

    px = pd.concat({"NTR": ntr, "MOS": mos}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 90
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
        return mark_failed("U7_russia_belarus_potash_rail", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U7 Belaruskali transit cut -> long NTR+MOS {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U7_russia_belarus_potash_rail", m, extra={
        "status": "ok",
        "rule": "When Belaruskali rail-transit through Lithuania falls > 20% YoY for 2 consecutive months (or equivalent sanction milestone), long NTR + MOS 50/50 for 90 trading days (~4.5 months); non-overlapping.",
        "mechanism": "Russia + Belarus = ~38% of mined potash. Klaipeda transit was Belaruskali's primary seaborne route; sanctions / transit halts tighten supply and re-rate NTR / MOS.",
        "source": "Curated Lithuanian Railways / press milestones (litrail.lt monthly stats not scraped). NTR + MOS via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~3-4); 2021-2023 sanctions cluster.",
    })


if __name__ == "__main__":
    main()
