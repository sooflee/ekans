"""
Q12 OPEC+ compliance rate -> Brent short signal.

IEA / Argus / S&P Global publish monthly OPEC+ compliance rates (headline figure -
percentage of pledged cuts actually delivered). When compliance falls below 80%
for 2 consecutive months, short BZ=F for 60 trading days.

Hardcoded headline compliance values (best-effort from public news summaries -
Reuters monthly OPEC reports, IEA OMR press, Platts). Values are approximate
percentages for OPEC+ overall compliance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)

# Monthly OPEC+ compliance rate (%), from press headlines / IEA OMR / Argus
# Note: months when no active cut deal -> mark None / 100 (irrelevant)
COMPLIANCE = {
    # 2020 - May deal effective
    "2020-05": 87, "2020-06": 110, "2020-07": 95, "2020-08": 102,
    "2020-09": 102, "2020-10": 96, "2020-11": 95, "2020-12": 99,
    # 2021
    "2021-01": 103, "2021-02": 113, "2021-03": 114, "2021-04": 114,
    "2021-05": 113, "2021-06": 113, "2021-07": 115, "2021-08": 116,
    "2021-09": 116, "2021-10": 115, "2021-11": 117, "2021-12": 122,
    # 2022 - underproduction (over-compliance)
    "2022-01": 132, "2022-02": 136, "2022-03": 157, "2022-04": 164,
    "2022-05": 161, "2022-06": 220, "2022-07": 218, "2022-08": 184,
    "2022-09": 198, "2022-10": 158, "2022-11": 169, "2022-12": 169,
    # 2023
    "2023-01": 161, "2023-02": 173, "2023-03": 170, "2023-04": 196,
    "2023-05": 173, "2023-06": 158, "2023-07": 165, "2023-08": 156,
    "2023-09": 157, "2023-10": 153, "2023-11": 110, "2023-12": 105,
    # 2024 - voluntary cuts; mixed
    "2024-01": 97, "2024-02": 96, "2024-03": 90, "2024-04": 85,
    "2024-05": 70, "2024-06": 74, "2024-07": 79, "2024-08": 76,
    "2024-09": 78, "2024-10": 84, "2024-11": 88, "2024-12": 92,
    # 2025
    "2025-01": 95, "2025-02": 92, "2025-03": 88, "2025-04": 85,
}


def main():
    try:
        bz = load_prices(["BZ=F"], start="2020-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("Q12_opec_compliance", f"BZ=F load failed: {e}")
    rets = bz.pct_change()

    s = pd.Series(COMPLIANCE)
    s.index = pd.to_datetime(s.index + "-01") + pd.offsets.MonthEnd(0)  # end-of-month
    s = s.sort_index()

    # detect 2 consecutive months < 80 - trigger on the 2nd month
    triggers = []
    keys = list(s.index)
    for i in range(1, len(keys)):
        if s.iloc[i] < 80 and s.iloc[i - 1] < 80:
            triggers.append(keys[i])

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    detail = []
    for d in triggers:
        # signal known at end of month; enter next session
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 60, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = -1.0  # short
        last_end = rets.index[end_idx - 1]
        n_events += 1
        detail.append(str(d.date()))

    if n_events == 0:
        return mark_failed("Q12_opec_compliance",
                           "no 2-consecutive-month <80% compliance episodes in hardcoded data")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q12 OPEC+ compliance<80% -> short BZ=F 60d")
    m["n_events"] = n_events
    print(f"Events: {n_events}; dates: {detail}")
    print_metrics(m)

    save_result("Q12_opec_compliance", m, extra={
        "status": "ok",
        "rule": "When OPEC+ headline compliance < 80% for 2 consecutive months (per IEA OMR / Argus headlines), short BZ=F for 60 trading days; non-overlapping.",
        "mechanism": "Sub-80% compliance signals cheating / quota collapse -> excess physical supply -> bearish Brent.",
        "source": "Hardcoded monthly compliance % from Reuters/Bloomberg/IEA OMR press summaries 2020-2025; yfinance BZ=F.",
        "n_events": n_events,
        "trigger_dates": detail,
    })


if __name__ == "__main__":
    main()
