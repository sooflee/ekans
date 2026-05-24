"""
U1 Indonesia PNBP nickel royalty hikes -> long Nickel Industries (NIC.AX).

Rule: When the Government of Indonesia raises the PNBP (Non-Tax State Revenue)
royalty rate on nickel ore / NPI / FeNi via a new PP / PMK regulation, long
NIC.AX for 30 trading days (~6 weeks). Non-overlapping.

Mechanism: Indonesia produces > 50% of mined nickel globally. PNBP royalty
hikes raise the marginal cost of NPI / FeNi from Indonesian RKAB-licensed
miners, tightening the cost curve and supporting nickel pricing. NIC.AX is a
pure-play Indonesian NPI / HPAL producer levered to nickel.

Source: ESDM / Kemenkeu Government Regulations (PP) and Minister of Finance
Regulations (PMK) - PP 26/2022, PMK 113/2024, PMK 26/2025 etc. Curated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated Indonesia nickel PNBP royalty-hike events.
# Date = publication / effective date of the relevant PP / PMK.
ROYALTY_EVENTS = [
    ("2020-06-29", "PP 81/2019 effective - first ad-valorem schedule"),
    ("2022-04-29", "PP 26/2022 - new royalty schedule with HPM-linked rates"),
    ("2024-10-23", "PMK 113/2024 announced higher tiered nickel ore royalty"),
    ("2025-04-26", "PMK 26/2025 - tiered royalty (10-19%) per HMA"),
    ("2025-07-15", "PMK supplemental hike - NPI / FeNi tier increase"),
]


def main():
    try:
        nic = load_prices(["NIC.AX"], start="2018-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("U1_indonesia_nickel_royalty", f"NIC.AX load failed: {e}")

    nic = nic.dropna()
    if len(nic) < 100:
        return mark_failed("U1_indonesia_nickel_royalty", "insufficient NIC.AX history")

    rets = nic.pct_change()
    HOLD = 30  # ~6 weeks

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    used = []
    for d_str, lbl in ROYALTY_EVENTS:
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
        return mark_failed("U1_indonesia_nickel_royalty", "no events landed in NIC.AX history")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U1 Indo PNBP nickel hike -> long NIC.AX {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U1_indonesia_nickel_royalty", m, extra={
        "status": "ok",
        "rule": "When the Government of Indonesia raises the PNBP nickel ore / NPI / FeNi royalty via PP or PMK, long NIC.AX for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "Indonesia produces > 50% of mined nickel; royalty hikes raise the marginal cost of Indonesian NPI / FeNi, supporting nickel pricing. NIC.AX is a levered pure-play.",
        "source": "Curated PP/PMK royalty-rate revision dates (PP 81/2019, PP 26/2022, PMK 113/2024, PMK 26/2025). NIC.AX via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N very small (~5); read with caution.",
    })


if __name__ == "__main__":
    main()
