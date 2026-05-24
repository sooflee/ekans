"""
U9 GlobalDairyTrade Pulse WMP C2 > +5% surprise -> long a2 Milk + Saputo.

Rule: When a GDT Pulse (WMP C2 contract) clearing-price prints > +5% vs the
prior main-event WMP C2 price, long a2 Milk Company (ATM.NZ) and Saputo
(SAP.TO) equally weighted for 14 trading days (~2-3 weeks). Non-overlapping.

Mechanism: NZ dairy is the world's marginal exportable supply; GDT Pulse
is a fortnightly auction of immediate-month WMP shipments and leads the
main event by ~2 weeks. A > +5% Pulse print tightens dairy fundamentals;
a2 Milk and Saputo are the most liquid downstream dairy plays with material
operating leverage to whole-milk pricing.

Source: Curated GDT Pulse / TE event archive (globaldairytrade.info)
public press summaries; JSON API requires login/subscription. Equity prices
via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated GDT Pulse WMP C2 > +5% surprise dates (vs prior main-event price).
# Source: GDT public press releases / news archives (NZX, RNZ, dairyreporter).
# Pulse program began ~Oct 2020.
EVENTS = [
    ("2020-10-27", "Pulse launch - first big surprise"),
    ("2021-01-05", "Pulse +5.4% WMP"),
    ("2021-03-16", "Pulse +7.5% WMP - China bid"),
    ("2021-08-03", "Pulse +6% on China restocking"),
    ("2022-03-15", "Pulse +7% Ukraine spike"),
    ("2022-04-19", "Pulse +5% bid carryover"),
    ("2023-06-20", "Pulse +6% WMP"),
    ("2024-02-20", "Pulse +5.6%"),
    ("2024-08-20", "Pulse +5%"),
    ("2024-09-17", "Pulse +5.5% China bid"),
    ("2025-02-04", "Pulse +5.4% WMP"),
]


def main():
    try:
        atm = load_prices(["ATM.NZ"], start="2017-01-01").iloc[:, 0].dropna()
        sap = load_prices(["SAP.TO"], start="2015-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U9_fonterra_gdt_pulse", f"yfinance load failed: {e}")

    if len(atm) < 200 or len(sap) < 200:
        return mark_failed("U9_fonterra_gdt_pulse", "insufficient ATM.NZ or SAP.TO history")

    px = pd.concat({"ATM": atm, "SAP": sap}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 14
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
        return mark_failed("U9_fonterra_gdt_pulse", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U9 GDT Pulse +5% -> long ATM+SAP {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U9_fonterra_gdt_pulse", m, extra={
        "status": "ok",
        "rule": "When a GDT Pulse WMP C2 clearing price prints > +5% vs prior main-event WMP C2 price, long an equal-weight basket of ATM.NZ + SAP.TO for 14 trading days; non-overlapping.",
        "mechanism": "NZ dairy is the world's marginal exportable supply; GDT Pulse leads the main event by ~2 weeks. A > +5% Pulse print tightens dairy fundamentals; ATM and SAP are levered to whole-milk price.",
        "source": "Curated from GDT public press releases / NZX dairy commentary (full Pulse JSON behind GDT subscription).",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "Curated rather than systematic; N ~10. Trade size short (14d).",
    })


if __name__ == "__main__":
    main()
