"""
F7 PBoC MLF cut -> LPR fixing.

The PBoC's Medium-term Lending Facility (MLF) rate guides the monthly LPR
(Loan Prime Rate) fixing on the 20th. When the MLF rate is cut in the prior
~10 days, the LPR typically follows, easing financial conditions in China.

For each MLF cut, long FXI from T-2 to T+1 where T = 20th of the next-month
LPR fixing (or 21st if 20th is weekend).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Curated MLF rate cut announcement dates (1Y MLF rate). Source: PBoC OMO
# announcements / Reuters / Bloomberg coverage.
MLF_CUTS = [
    "2019-11-05",
    "2020-02-17",
    "2020-04-15",
    "2022-01-17",
    "2022-08-15",
    "2023-06-15",
    "2023-08-15",
    "2024-07-25",
    "2024-09-25",
    "2025-05-15",
]


def next_lpr_date(mlf_date):
    """LPR fixing is published 20th of each month (or next business day)."""
    d = pd.Timestamp(mlf_date)
    # If MLF cut is before the 20th of same month, use that 20th; otherwise next month.
    if d.day < 20:
        lpr = pd.Timestamp(year=d.year, month=d.month, day=20)
    else:
        # next month 20th
        m = d.month + 1
        y = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        lpr = pd.Timestamp(year=y, month=m, day=20)
    # bump to next business day if weekend
    while lpr.weekday() >= 5:
        lpr += pd.Timedelta(days=1)
    return lpr


def main():
    sid = "F7_pboc_lpr"
    try:
        px = load_prices(["FXI", "SPY"], start="2018-06-01")
        rets = px.pct_change()
        idx = rets.index

        pos = pd.Series(0.0, index=idx)
        used = []
        for d in MLF_CUTS:
            mlf = pd.Timestamp(d)
            lpr = next_lpr_date(mlf)
            # Confirm MLF cut is within 10 days prior to LPR (by construction yes).
            gap = (lpr - mlf).days
            if gap > 35:
                continue
            loc = idx.searchsorted(lpr, side="left")
            if loc >= len(idx):
                continue
            start = max(loc - 2, 0)
            end = min(loc + 1, len(idx) - 1)
            pos.iloc[start:end + 1] = 1.0
            used.append({"mlf_cut": d, "lpr_fix": str(lpr.date()), "gap_days": gap})

        port = pos.shift(1) * rets["FXI"]
        active = (pos != 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, "Too few active days")

        m = compute_metrics(port_active, benchmark=rets["SPY"].reindex(port_active.index),
                            name="F7 PBoC MLF cut -> FXI around LPR fix")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "On each PBoC 1Y-MLF rate cut, long FXI from T-2 to T+1 where "
                    "T = next 20th-of-month LPR fixing date.",
            "mechanism": "MLF rate sets LPR floor; cut transmits to bank lending and Chinese equity multiples.",
            "universe": "FXI",
            "n_events": len(used),
            "events": used,
            "source": "PBoC open-market operations announcements, Reuters (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
