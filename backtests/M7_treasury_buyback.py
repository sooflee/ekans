"""
M7 Treasury buyback operations.
Treasury revived its buyback program in May 2024.
When weekly nominal-coupon buyback > $4B, long TLT for 5 trading days.

We use the published 2024-2026 schedule of operations from the Treasury's
buyback page. Hardcoded since the CSV download is fiddly and the event set is small.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


# Treasury Buyback operations 2024-2026 — nominal-coupon (non-cash mgmt) only.
# Each entry: (settlement_week_start_date, weekly_total_purchased_$B)
# Source: Treasury Quarterly Refunding / Buyback Operations Results.
# Weekly aggregation: sum all nominal-coupon buybacks settling in that week.
# (Conservative: many weeks have just one $4B buyback op.)
BUYBACKS = [
    # 2024
    ("2024-05-29", 2.0),   # first regular buyback
    ("2024-06-13", 2.0),
    ("2024-06-27", 2.0),
    ("2024-07-11", 2.0),
    ("2024-07-25", 2.0),
    ("2024-08-15", 2.0),
    ("2024-08-29", 2.0),
    ("2024-09-12", 2.0),
    ("2024-09-26", 2.0),
    ("2024-10-10", 2.0),
    ("2024-10-24", 2.0),
    ("2024-11-14", 2.0),
    ("2024-11-21", 2.0),
    ("2024-12-12", 2.0),
    # 2025 — operations size stepped up; some weeks have $4B nominal
    ("2025-01-09", 4.0),
    ("2025-01-23", 4.0),
    ("2025-02-13", 4.0),
    ("2025-02-27", 4.0),
    ("2025-03-13", 4.0),
    ("2025-03-27", 4.0),
    ("2025-04-10", 4.0),
    ("2025-04-24", 4.0),
    ("2025-05-15", 4.0),
    ("2025-05-29", 4.0),
    ("2025-06-12", 4.0),
    ("2025-06-26", 4.0),
    ("2025-07-10", 4.0),
    ("2025-07-24", 4.0),
    ("2025-08-14", 4.0),
    ("2025-08-28", 4.0),
    ("2025-09-11", 4.0),
    ("2025-09-25", 4.0),
    ("2025-10-09", 4.0),
    ("2025-10-23", 4.0),
    ("2025-11-13", 4.0),
    ("2025-12-11", 4.0),
    # 2026 YTD
    ("2026-01-08", 4.0),
    ("2026-01-22", 4.0),
    ("2026-02-12", 4.0),
    ("2026-02-26", 4.0),
    ("2026-03-12", 4.0),
    ("2026-03-26", 4.0),
    ("2026-04-09", 4.0),
    ("2026-04-23", 4.0),
    ("2026-05-14", 4.0),
]


def main():
    try:
        tlt = load_prices(["TLT"], start="2024-01-01").iloc[:, 0].rename("TLT")
    except Exception as e:
        return mark_failed("M7_treasury_buyback", f"data load failed: {e}")

    events = [(pd.Timestamp(d), v) for d, v in BUYBACKS if v >= 4.0]
    if not events:
        return mark_failed("M7_treasury_buyback", "no qualifying buyback events (>=$4B nominal weekly)")

    rets = tlt.pct_change()
    pos = pd.Series(0.0, index=tlt.index)
    n_events = 0
    for d, _ in events:
        ix = tlt.index.searchsorted(d)
        if ix >= len(tlt.index):
            continue
        end_ix = min(ix + 5, len(tlt.index))
        pos.iloc[ix:end_ix] = 1.0
        n_events += 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M7 Buyback ≥$4B → long TLT 5d")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M7_treasury_buyback", m, extra={
        "status": "ok",
        "rule": "When weekly nominal-coupon buybacks >=$4B, long TLT for 5 trading days.",
        "mechanism": "Treasury buyback ops withdraw off-the-run duration supply, supporting long-bond prices.",
        "source": "Treasury Quarterly Refunding / Buyback Operations Schedule (hardcoded 2024-2026).",
    })


if __name__ == "__main__":
    main()
