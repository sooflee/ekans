"""
S13 California April-1 statewide snowpack -> long DC=F (milk futures).

Rule: When statewide California snowpack (Snow Water Equivalent, % of April-1
historical average) prints below 70% on the April-1 snow survey, long DC=F
(Class III milk futures) for ~126 trading days (~6 months). Non-overlapping
events.

Mechanism: Low Sierra snowpack -> reduced summer irrigation deliveries to the
Central Valley -> hay / alfalfa cost inflation + heat-stressed cow productivity
-> Class III milk futures rally over the summer months.

Source: California Department of Water Resources / CDEC official April-1
snowpack press releases ("Bulletin 120" / Statewide Snow Survey results).
Values are statewide average % of April-1 historical mean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# CDEC / DWR Statewide April-1 SWE % of April-1 average (Apr 1 measurement)
# Sources: CA DWR press releases; CDEC Snow Survey historical bulletins.
APR1_SWE_PCT = [
    ("2000-04-01", 102),
    ("2001-04-01",  87),
    ("2002-04-01", 116),
    ("2003-04-01", 121),
    ("2004-04-01",  90),
    ("2005-04-01", 156),
    ("2006-04-01", 173),
    ("2007-04-01",  46),
    ("2008-04-01",  81),
    ("2009-04-01",  81),
    ("2010-04-01", 116),
    ("2011-04-01", 168),
    ("2012-04-01",  55),
    ("2013-04-01",  52),
    ("2014-04-01",  25),
    ("2015-04-01",   5),
    ("2016-04-01",  85),
    ("2017-04-01", 158),
    ("2018-04-01",  52),
    ("2019-04-01", 162),
    ("2020-04-01",  53),
    ("2021-04-01",  60),
    ("2022-04-01",  38),
    ("2023-04-01", 237),
    ("2024-04-01", 110),
    ("2025-04-01",  90),
]


def main():
    try:
        dc = load_prices(["DC=F"], start="2005-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S13_ca_snowpack", f"DC=F load failed: {e}")

    rets = dc.pct_change()
    if len(rets.dropna()) < 200:
        return mark_failed(
            "S13_ca_snowpack",
            f"DC=F free-data series too thin / illiquid ({rets.dropna().shape[0]} returns); "
            "milk futures are typically thinly quoted on Yahoo with many no-trade days.",
            extra={"apr1_records": len(APR1_SWE_PCT)},
        )

    triggers = [(pd.Timestamp(d), v) for d, v in APR1_SWE_PCT if v < 70]
    print(f"April-1 records: {len(APR1_SWE_PCT)}; <70% triggers: {len(triggers)}")
    for d, v in triggers:
        print(f"  {d.date()}: {v}%")

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
        return mark_failed("S13_ca_snowpack",
                           "no April-1 SWE < 70% events landed on DC=F trading dates")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S13 CA Apr-1 SWE <70% -> long DC=F 126d")
    m["n_events"] = n_events
    print_metrics(m)

    save_result("S13_ca_snowpack", m, extra={
        "status": "ok",
        "rule": "When California statewide April-1 snowpack SWE prints < 70% of historical April-1 average, long DC=F (CME Class III milk futures front-month) for 126 trading days; non-overlapping events.",
        "mechanism": "Low snowpack -> reduced summer Central Valley irrigation -> alfalfa & hay cost inflation + heat-stressed cows -> Class III milk futures rally into summer.",
        "source": "CA DWR / CDEC Snow Survey April-1 statewide bulletins (cdec.water.ca.gov). DC=F via yfinance (note: free data is illiquid).",
        "n_events": n_events,
        "first_events": event_dates,
    })


if __name__ == "__main__":
    main()
