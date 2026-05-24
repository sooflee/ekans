"""
A11 DST anomaly (Kamstra-Kramer-Levi 2000).
Short SPY only on the Monday after each DST change in the US:
  - Spring forward: 2nd Sunday of March
  - Fall back: 1st Sunday of November
(Post-2007 schedule applies; prior to 2007 it was 1st Sun Apr / last Sun Oct.
US Energy Policy Act 2005 effective 2007.)

Report mean return on DST Mondays vs all-Mondays baseline; report t-stat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime as dt
import calendar
import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result,
)


def nth_weekday_of_month(year, month, n, weekday):
    """n-th occurrence of weekday (Mon=0..Sun=6) in (year, month). Returns date."""
    days = [d for d in calendar.Calendar().itermonthdates(year, month)
            if d.month == month and d.weekday() == weekday]
    return days[n - 1]


def last_weekday_of_month(year, month, weekday):
    days = [d for d in calendar.Calendar().itermonthdates(year, month)
            if d.month == month and d.weekday() == weekday]
    return days[-1]


def dst_dates(year):
    """Return (spring_sunday, fall_sunday) for the US schedule applicable in `year`."""
    if year >= 2007:
        spring = nth_weekday_of_month(year, 3, 2, 6)   # 2nd Sunday March
        fall = nth_weekday_of_month(year, 11, 1, 6)    # 1st Sunday November
    else:
        spring = nth_weekday_of_month(year, 4, 1, 6)   # 1st Sunday April
        fall = last_weekday_of_month(year, 10, 6)      # last Sunday October
    return spring, fall


def next_monday(d):
    days_ahead = (0 - d.weekday()) % 7  # Mon=0
    if days_ahead == 0:
        days_ahead = 7
    return d + dt.timedelta(days=days_ahead)


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    # Build set of DST-Monday trading dates
    dst_mondays = []
    for y in range(idx[0].year, idx[-1].year + 1):
        spring, fall = dst_dates(y)
        for sunday in (spring, fall):
            mon = next_monday(sunday)
            # snap to next trading day if Mon not trading
            i = idx.searchsorted(pd.Timestamp(mon), side="left")
            if i < len(idx):
                # within 3 days tolerance, treat as DST-Monday
                d_actual = idx[i]
                if (d_actual.date() - mon).days <= 3:
                    dst_mondays.append(d_actual)

    dst_set = pd.DatetimeIndex(dst_mondays)
    # Strategy: short SPY on DST Mondays only (close T-1 to close T).
    # Position at T-1 close = -1 to earn -ret(T). i.e., pos[d-1] = -1 for d in dst_set.
    pos = pd.Series(0.0, index=idx)
    for d in dst_set:
        i = idx.get_loc(d)
        if i > 0:
            pos.iloc[i - 1] = -1.0

    pnl = pos.shift(1) * rets
    pnl = pnl.dropna()

    # Compute additional event-study stats
    dst_ret = rets.loc[dst_set.intersection(rets.index)]
    all_mondays = rets[rets.index.weekday == 0]
    t_stat_dst = dst_ret.mean() / (dst_ret.std() / np.sqrt(len(dst_ret))) if dst_ret.std() > 0 else float('nan')

    m = compute_metrics(pnl, benchmark=rets, name="A11 DST short-Monday")
    print_metrics(m)
    print(f"  DST-Monday mean return: {dst_ret.mean()*100:.4f}%  (n={len(dst_ret)})")
    print(f"  All-Monday mean return: {all_mondays.mean()*100:.4f}%  (n={len(all_mondays)})")
    print(f"  DST-Monday t-stat:      {t_stat_dst:.2f}")
    save_result("A11_dst_anomaly", m, extra={
        "status": "ok",
        "rule": "Short SPY only on Monday after each US DST change.",
        "universe": "SPY",
        "source": "Kamstra-Kramer-Levi (AER 2000)",
        "n_events": len(dst_set),
        "dst_monday_mean_ret": float(dst_ret.mean()),
        "all_monday_mean_ret": float(all_mondays.mean()),
        "dst_monday_t_stat": float(t_stat_dst),
    })


if __name__ == "__main__":
    main()
