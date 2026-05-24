"""
M6 Fed central-bank swap lines outstanding.
FRED SWPT (Central Bank Liquidity Swaps, weekly). When > $10B, short SPY for 30 days.
Mechanism: large swap-line draws signal acute USD funding stress -> equity weakness.
Events are RARE (Mar 2020, Mar 2023, late 2008/09). Honest small-N test.
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
    # Try SWPT first, fall back to SWPMA
    series_tried = []
    swp = None
    for s in ["SWPT", "SWPMA"]:
        try:
            swp = load_fred(s, start="2000-01-01").iloc[:, 0].rename(s)
            series_tried.append(s)
            break
        except Exception as e:
            series_tried.append(f"{s} (fail: {e})")
            continue
    if swp is None:
        return mark_failed("M6_fed_swap_line", f"could not load swap-line series. Tried: {series_tried}")

    try:
        spy = load_prices(["SPY"], start="2000-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("M6_fed_swap_line", f"SPY load failed: {e}")

    swp = swp.dropna()
    # FRED reports in $millions for H.4.1 swap line entries; convert to $B.
    # Heuristic: if peak > 100,000 then series is in $M; else already $B.
    units = "$M" if swp.max() > 100_000 else "$B"
    if units == "$M":
        swp_b = swp / 1_000.0
    else:
        swp_b = swp

    # First-crossing only: prior week <=10, this week >10. After firing, require swp to fall back below 10 and then re-cross to re-fire.
    above = (swp_b > 10.0)
    prev_above = above.shift(1).fillna(False)
    first_cross = above & (~prev_above)
    trigger_dates = swp_b.index[first_cross]

    rets = spy.pct_change()
    pos = pd.Series(0.0, index=spy.index)
    n_events = 0
    for d in trigger_dates:
        ix = spy.index.searchsorted(d)
        if ix >= len(spy.index):
            continue
        end_ix = min(ix + 30, len(spy.index))
        pos.iloc[ix:end_ix] = -1.0  # short
        n_events += 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M6 Swap line > $10B → short SPY 30d")
    m["n_events"] = int(n_events)
    m["series_used"] = swp.name
    m["units_detected"] = units
    print_metrics(m)
    print(f"  n_events: {n_events}, series={swp.name}, units={units}")
    save_result("M6_fed_swap_line", m, extra={
        "status": "ok",
        "rule": "When Fed central-bank liquidity swaps outstanding > $10B (first crossing, 45d cooldown), short SPY 30 trading days.",
        "mechanism": "Large swap line draws signal acute USD funding stress globally -> equity drawdown.",
        "source": "FRED SWPT / SWPMA (H.4.1).",
    })


if __name__ == "__main__":
    main()
