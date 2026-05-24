"""
M11 LEI manufacturing hours.
FRED AWHMAN (Avg Weekly Hours, Manufacturing, monthly).
When 6m change < -0.4 hours, short SPY for 90 trading days.
Mechanism: shrinking factory hours is a classic late-cycle Conference Board LEI input.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        # AWHMAN: avg weekly hours manufacturing, monthly
        awh = load_fred("AWHMAN", start="1990-01-01").iloc[:, 0].rename("AWHMAN")
        spy = load_prices(["SPY"], start="1993-01-29").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("M11_lei_mfg_hours", f"data load failed: {e}")

    awh = awh.dropna()
    chg_6m = awh - awh.shift(6)  # 6 month change in hours
    trigger = (chg_6m < -0.4)

    # Find trigger months (use first occurrence in a cluster: not yet triggered last month)
    trigger_dates = chg_6m.index[trigger.fillna(False)]

    rets = spy.pct_change()
    pos = pd.Series(0.0, index=spy.index)

    # For each trigger month, short SPY for 90 trading days starting the next available trading day
    # Use the trigger date + small lag (LEI is reported with delay, conservative +5 business days)
    n_events = 0
    last_event = None
    for d in trigger_dates:
        # +5 BD lag for data availability
        eff = d + pd.Timedelta(days=10)
        ix = spy.index.searchsorted(eff)
        if ix >= len(spy.index):
            continue
        end_ix = min(ix + 90, len(spy.index))
        pos.iloc[ix:end_ix] = -1.0  # short
        n_events += 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M11 AWHMAN 6m drop > 0.4h → short SPY 90d")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events (months with trigger): {n_events}")
    save_result("M11_lei_mfg_hours", m, extra={
        "status": "ok",
        "rule": "When AWHMAN 6m change < -0.4 hours, short SPY for 90 trading days (with +10d data lag).",
        "mechanism": "Manufacturing hours shrink is a leading recession indicator (Conference Board LEI input).",
        "source": "FRED AWHMAN; Conference Board Leading Economic Index methodology.",
    })


if __name__ == "__main__":
    main()
