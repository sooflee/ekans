"""
U5 Madagascar SAVA-region cyclones -> long Symrise (SY1.DE) + McCormick (MKC).

Rule: When a named tropical cyclone makes landfall in the SAVA region (NE
Madagascar - Sambava / Antalaha / Vohemar / Andapa, ~80% of world vanilla)
with measurable crop damage, long Symrise (SY1.DE) and McCormick (MKC)
equally weighted for 126 trading days (~6 months). Non-overlapping.

Mechanism: Madagascar supplies ~80% of natural-vanilla beans; SAVA cyclone
damage tightens supply 6-18 months out as flower-stage trees suffer; vanilla
prices rise; Symrise (largest vanilla extractor, 'Aust & Hachmann' subsidiary)
benefits via mark-to-market on inventory; McCormick passes pricing.

Source: NASA / WMO RSMC La Reunion cyclone database (curated by name and
SAVA-landfall flag). SY1.DE and MKC via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated SAVA-landfall named cyclones with documented vanilla-crop damage.
EVENTS = [
    ("2017-03-07", "Cyclone Enawo (Cat 4) makes landfall at Antalaha"),
    ("2020-01-26", "Cyclone Diane (TS) hits NE Madagascar"),
    ("2022-02-05", "Cyclone Batsirai (Cat 3) - SAVA damage"),
    ("2022-02-22", "Cyclone Emnati - residual SAVA damage"),
    ("2023-02-22", "Cyclone Freddy (Cat 4) - SE then SAVA"),
    ("2024-03-25", "Cyclone Gamane - direct SAVA landfall"),
]


def main():
    try:
        sy = load_prices(["SY1.DE"], start="2010-01-01").iloc[:, 0].dropna()
        mkc = load_prices(["MKC"], start="2010-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U5_madagascar_cyclone_vanilla", f"yfinance load failed: {e}")

    if len(sy) < 200 or len(mkc) < 200:
        return mark_failed("U5_madagascar_cyclone_vanilla", "insufficient SY1.DE/MKC history")

    px = pd.concat({"SY1": sy, "MKC": mkc}, axis=1).sort_index()
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
        return mark_failed("U5_madagascar_cyclone_vanilla", "no events landed in price history")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U5 SAVA cyclone -> long SY1+MKC {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U5_madagascar_cyclone_vanilla", m, extra={
        "status": "ok",
        "rule": "When a named cyclone makes SAVA-region (NE Madagascar) landfall with crop damage, long equal-weight Symrise (SY1.DE) + McCormick (MKC) for 126 trading days (~6 months); non-overlapping.",
        "mechanism": "Madagascar supplies ~80% of natural vanilla; SAVA cyclone damage tightens supply 6-18 months out and lifts vanilla prices, benefiting downstream extractors and packaged-food pricing power.",
        "source": "RSMC La Reunion cyclone track database, cross-checked with WMO / press reports. Equity prices via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~6); Symrise / McCormick have broad consumer-staples beta.",
    })


if __name__ == "__main__":
    main()
