"""
S4 USDA October Florida Citrus Forecast cuts -> long OJ=F.

Rule: USDA NASS releases its first official Florida orange production forecast
in October each year. When the October forecast is > 15% below the prior
season's final production, long OJ=F front-month for ~126 trading days
(~6 months). Non-overlapping events.

Mechanism: A material October cut signals smaller crop already locked in
(citrus greening, hurricane damage, cold-snap stress) and historically
forces a forward-supply repricing in OJ futures into the post-harvest period.

Source: USDA NASS Florida Citrus Forecast monthly PDF (October releases) -
values curated from official NASS press releases (citrusforecast.nass.usda.gov).
Production figures in million 90-lb boxes, all-orange total (early/mid + Valencia).

Values represent the OCTOBER forecast year (Y) vs prior-season FINAL (Y-1 final
from the post-July of next year). Sources cross-referenced with
nass.usda.gov 'Citrus Fruits Annual Summary' historical reports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# (Oct forecast release date, October forecast in MM 90lb boxes, prior-season FINAL)
# Curated from NASS Florida Citrus Forecast / Citrus Fruits Annual Summary press releases.
# Note: cumulative cuts due to citrus greening (HLB) + hurricanes Irma (2017), Ian (2022)
# drove repeated step-downs.
FORECASTS = [
    # release_date    Oct forecast (Mboxes), prior-year final (Mboxes), season label
    ("2010-10-12", 145.5, 133.6, "2010-11"),
    ("2011-10-12", 150.0, 140.2, "2011-12"),  # actual: 146.7
    ("2012-10-11", 154.0, 146.7, "2012-13"),
    ("2013-10-11", 125.0, 133.6, "2013-14"),  # HLB stepdown
    ("2014-10-10", 108.0, 104.4, "2014-15"),
    ("2015-10-09",  80.0,  96.8, "2015-16"),  # -17% large cut
    ("2016-10-12",  70.0,  81.5, "2016-17"),  # -14% borderline
    ("2017-10-12",  54.0,  68.7, "2017-18"),  # Hurricane Irma -21%
    ("2018-10-11",  79.0,  44.9, "2018-19"),  # recovery (positive)
    ("2019-10-10",  74.0,  71.7, "2019-20"),
    ("2020-10-09",  57.0,  67.4, "2020-21"),  # -15% borderline
    ("2021-10-12",  47.0,  52.4, "2021-22"),
    ("2022-10-12",  28.0,  41.2, "2022-23"),  # Hurricane Ian -32%
    ("2023-10-12",  20.5,  15.8, "2023-24"),
    ("2024-10-11",  15.0,  17.5, "2024-25"),
]


def main():
    try:
        oj = load_prices(["OJ=F"], start="2009-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S4_florida_citrus", f"OJ=F load failed: {e}")

    rets = oj.pct_change()

    triggers = []
    for rel_str, fcast, prior, season in FORECASTS:
        rel = pd.Timestamp(rel_str)
        cut_pct = (fcast / prior) - 1.0
        if cut_pct < -0.15:
            triggers.append((rel, cut_pct, season))

    print(f"Total Oct forecasts: {len(FORECASTS)}; cuts > 15%: {len(triggers)}")
    for d, p, s in triggers:
        print(f"  {d.date()} {s}: {p*100:+.1f}%")

    HOLD = 126
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d, _, _ in triggers:
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
        return mark_failed("S4_florida_citrus", "no Oct forecast cuts > 15% in window")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S4 FL Citrus Oct -15% cut -> long OJ=F 126d")
    m["n_events"] = n_events
    print_metrics(m)

    save_result("S4_florida_citrus", m, extra={
        "status": "ok",
        "rule": "When USDA NASS October Florida Citrus Forecast (all-orange production, MM 90-lb boxes) is > 15% below the prior season's FINAL production, long OJ=F front-month next session for 126 trading days (~6 months); non-overlapping events.",
        "mechanism": "An October cut crystallises a smaller forward crop (citrus greening, hurricanes, freezes) and triggers a multi-month supply repricing in OJ futures.",
        "source": "Curated from USDA NASS Florida Citrus Forecast (citrusforecast.nass.usda.gov) October press releases and Citrus Fruits Annual Summary. OJ=F via yfinance. Spec mentions JO ETF (iPath Bloomberg OJ) which is delisted on Yahoo; OJ=F front-month used instead.",
        "n_events": n_events,
        "first_events": event_dates,
        "triggers": [(str(d.date()), round(p, 4), s) for d, p, s in triggers],
    })


if __name__ == "__main__":
    main()
