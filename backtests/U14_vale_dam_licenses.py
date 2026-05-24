"""
U14 Vale ANM SIGBM dam-stability downgrade events -> long FMG (iron-ore proxy)
   or short VALE.

Rule: When Brazil's ANM SIGBM database records a DCE / DCO downgrade for a
Vale tailings dam, long Fortescue (FMG.AX) for 30 trading days (~6 weeks).
Non-overlapping.

Mechanism: Vale produces ~17% of seaborne iron-ore. Stability-credential
downgrades force evacuation of dam sub-watersheds, sometimes shutting
adjacent mine complexes (Brucutu, Vargem Grande). Seaborne supply tightens;
FMG (Australia, low-cost) is the cleanest beneficiary; VALE itself
sometimes sells off near-term and recovers later.

Source: ANM SIGBM is officially listed but historical scrape is heavy.
Curated DCE / DCO Vale downgrade dates from press archives.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated Vale dam-stability downgrade events (DCE = nao garantia, DCO = nivel emergencia).
EVENTS = [
    ("2019-01-25", "Brumadinho B1 dam collapse (catalyst)"),
    ("2019-02-05", "Vale halts Brucutu - PA-MA dam"),
    ("2019-03-22", "Sul Superior - DCE downgrade"),
    ("2019-04-22", "Vargem Grande complex shut"),
    ("2020-02-21", "Forquilha III - level 2 evacuation"),
    ("2021-04-14", "Doutor / Xingu DCE downgrade"),
    ("2022-04-08", "Sul Superior level 3 maintained"),
    ("2023-03-15", "Mina do Meio / Aboboras DCE-related"),
]


def main():
    try:
        fmg = load_prices(["FMG.AX"], start="2014-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U14_vale_dam_licenses", f"FMG.AX load failed: {e}")

    if len(fmg) < 200:
        return mark_failed("U14_vale_dam_licenses", "insufficient FMG.AX history")

    rets = fmg.pct_change()
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
        return mark_failed("U14_vale_dam_licenses", "no events landed")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U14 Vale dam downgrade -> long FMG {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U14_vale_dam_licenses", m, extra={
        "status": "ok",
        "rule": "When Brazil's ANM SIGBM records a DCE / DCO downgrade for a Vale tailings dam, long FMG.AX for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "Vale = ~17% of seaborne iron-ore; stability-credential downgrades force evacuation and adjacent-mine shut-ins. FMG (Australia, low-cost) is the cleanest seaborne beneficiary.",
        "source": "Curated DCE/DCO downgrade dates from press archives (Reuters, Valor) - ANM SIGBM historical scrape not implemented. FMG.AX via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N moderate (~8); 2019 Brumadinho cluster dominates effect size.",
    })


if __name__ == "__main__":
    main()
