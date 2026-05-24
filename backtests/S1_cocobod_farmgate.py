"""
S1 Ghana COCOBOD farmgate price hikes -> long cocoa via CC=F.

Rule: When Ghana COCOBOD announces a farmgate cocoa-price hike of > 20%,
long CC=F (NY cocoa) for ~126 trading days (6 months). Non-overlapping events.

(Spec specified NIB iPath cocoa ETN, which is delisted on Yahoo. Substituting
CC=F front-month, which is the underlying for NIB.)

Mechanism: COCOBOD farmgate hikes are an official Ghana / Cote d'Ivoire policy
signal acknowledging local supply tightness (smuggling concerns, deficit
seasons, currency weakness). Hikes >20% historically follow a global cocoa
deficit and tend to coincide with multi-month rallies in CC=F.

Source: Curated from Reuters / Bloomberg COCOBOD press release coverage and
COCOBOD news section (cocobod.gh/news). Farmgate prices reported in GHS per
64-kg bag; the percentage change is computed in nominal local currency, which
is what COCOBOD itself communicates and what the market reacts to.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# COCOBOD producer-price announcements: (announce_date, new_price GHS/bag, prior_price GHS/bag)
# Curated from Reuters / Bloomberg coverage and cocobod.gh news.
HIKES = [
    ("2010-10-01", 200.00, 160.00),  # +25%
    ("2011-10-12", 205.00, 200.00),  # ~+2.5% (no hike)
    ("2012-10-01", 212.00, 205.00),  # ~+3.4% (no hike)
    ("2013-10-04", 224.00, 212.00),  # +5.7%
    ("2014-10-01", 345.00, 224.00),  # +54% — major hike
    ("2015-10-01", 420.00, 345.00),  # +22% — hike
    ("2016-10-13", 475.00, 420.00),  # +13%
    ("2017-10-02", 475.00, 475.00),  # 0%
    ("2018-10-01", 475.00, 475.00),  # 0%
    ("2019-10-01", 515.00, 475.00),  # +8.4%
    ("2020-10-01", 660.00, 515.00),  # +28% (LID + politics)
    ("2021-10-01", 660.00, 660.00),  # 0%
    ("2022-10-13", 800.00, 660.00),  # +21% — hike
    ("2023-09-29", 1308.00, 800.00),  # +63% — major
    ("2024-04-05", 2400.00, 1308.00),  # +83% — extraordinary
    ("2024-09-12", 3000.00, 2400.00),  # +25% — hike
]


def main():
    try:
        cc = load_prices(["CC=F"], start="2008-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S1_cocobod_farmgate", f"CC=F load failed: {e}")

    rets = cc.pct_change()

    triggers = []
    for d, new_p, old_p in HIKES:
        chg = new_p / old_p - 1
        if chg > 0.20:
            triggers.append((pd.Timestamp(d), chg))

    print(f"COCOBOD announcements: {len(HIKES)}; hikes > 20%: {len(triggers)}")
    for d, p in triggers:
        print(f"  {d.date()}: {p*100:+.1f}%")

    HOLD = 126
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d, _ in triggers:
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
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S1_cocobod_farmgate", "no COCOBOD hikes > 20% in curated history")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S1 COCOBOD farmgate +>20% -> long CC=F 126d")
    m["n_events"] = n_events
    print_metrics(m)

    save_result("S1_cocobod_farmgate", m, extra={
        "status": "ok",
        "rule": "When Ghana COCOBOD announces a farmgate producer price hike > +20% (nominal GHS/bag), long CC=F next session for 126 trading days (~6 months); non-overlapping events.",
        "mechanism": "COCOBOD hikes are an official policy signal of West African supply tightness (smuggling pressure, deficit seasons, FX weakness). Hikes > 20% historically follow global cocoa deficits and coincide with multi-month rallies in NY cocoa futures.",
        "source": "Curated from Reuters / Bloomberg COCOBOD coverage and cocobod.gh news section. CC=F via yfinance (NIB iPath ETN is delisted on Yahoo).",
        "n_events": n_events,
        "first_events": event_dates,
        "substitution_note": "NIB delisted -> using CC=F front-month directly.",
    })


if __name__ == "__main__":
    main()
