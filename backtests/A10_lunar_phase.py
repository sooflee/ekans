"""
A10 Lunar phase.
Compute full-moon and new-moon dates programmatically.
Test: long SPY only on new-moon ± 7 day windows, short during full-moon ± 7 windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import datetime as dt
import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def moon_phase_fraction(date):
    """Return illuminated fraction of the moon for `date` (datetime.date).
    Source: Conway approximation. 0 = new moon, 1 = full moon.
    """
    # Days since known new moon: 2000-01-06 18:14 UTC
    ref = dt.datetime(2000, 1, 6, 18, 14)
    d = dt.datetime(date.year, date.month, date.day) - ref
    synodic = 29.530588853
    age = (d.total_seconds() / 86400.0) % synodic
    # Illumination = (1 - cos(2pi * age/synodic)) / 2
    return 0.5 * (1 - math.cos(2 * math.pi * age / synodic))


def find_phase_dates(start, end):
    """Yield (date, phase_name) for each new moon and full moon between start and end.
    Detect by sign change of (illumination - 0.5) or by detecting the local min/max
    of illumination. We'll detect new moon (illum local minimum near 0) and full moon
    (local maximum near 1) by scanning per day.
    """
    cur = start
    new_moons = []
    full_moons = []
    illum_prev2, illum_prev = None, None
    date_prev2, date_prev = None, None
    while cur <= end:
        illum = moon_phase_fraction(cur)
        if illum_prev is not None and illum_prev2 is not None:
            # Local min => new moon
            if illum_prev < illum_prev2 and illum_prev < illum:
                new_moons.append(date_prev)
            # Local max => full moon
            if illum_prev > illum_prev2 and illum_prev > illum:
                full_moons.append(date_prev)
        illum_prev2, illum_prev = illum_prev, illum
        date_prev2, date_prev = date_prev, cur
        cur = cur + dt.timedelta(days=1)
    return new_moons, full_moons


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    new_moons, full_moons = find_phase_dates(idx[0].date(), idx[-1].date())

    pos = pd.Series(0.0, index=idx)
    window = 7  # ±7 calendar days

    for nm in new_moons:
        lo = pd.Timestamp(nm) - pd.Timedelta(days=window)
        hi = pd.Timestamp(nm) + pd.Timedelta(days=window)
        mask = (idx >= lo) & (idx <= hi)
        pos.loc[mask] = pos.loc[mask] + 1.0  # add long signal

    for fm in full_moons:
        lo = pd.Timestamp(fm) - pd.Timedelta(days=window)
        hi = pd.Timestamp(fm) + pd.Timedelta(days=window)
        mask = (idx >= lo) & (idx <= hi)
        pos.loc[mask] = pos.loc[mask] - 1.0  # add short signal

    # Cap at [-1, +1]: full-moon and new-moon windows overlap for ~15-day cycle, so
    # most days will net to 0. The non-zero days are the "pure" windows near a phase.
    pos = pos.clip(-1, 1)

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A10 Lunar phase")
    print_metrics(m)
    save_result("A10_lunar_phase", m, extra={
        "status": "ok",
        "rule": "Long ±7d around new moons, short ±7d around full moons; signals clipped to [-1,1] on overlap.",
        "universe": "SPY",
        "source": "Dichev-Janes 2003; Yuan-Zheng-Zhu 2006",
        "n_new_moons": len(new_moons),
        "n_full_moons": len(full_moons),
    })


if __name__ == "__main__":
    main()
