"""
T8 Eskom load shedding -> long platinum (PL=F).

Rule: When Eskom Load-Shedding Stage >= 4 for 7+ consecutive days,
go long PL=F for 2 months (~42 trading days).

Data substitution:
  - EskomSePush API requires a paid token (free tier rate-limited and lacks
    historical archive). Public Eskom JSON only returns "current status".
  - We curate event start dates from public news reporting:
       https://en.wikipedia.org/wiki/South_African_energy_crisis
       Daily Maverick, News24 archives, EskomSePush blog posts.
    Each event is a date when Stage 4+ persisted for >= 7 consecutive days.

Curated event list (start dates of qualifying Stage 4+ multi-week episodes):
  2019-02-11   Stage 4 declared for first time in ~3 years, lasted ~14 days
  2019-12-09   Stage 6 first time, persistent Stage 4-6 for >2 weeks
  2022-06-21   Persistent Stage 4-6, multi-week
  2022-09-18   Stage 6 again, multi-week
  2022-12-01   Stage 6 -> ongoing, crisis level all winter
  2023-01-15   Continuous Stage 4-6 (during winter; crisis peak Q1 2023)
  2023-09-15   Resurgence ahead of summer
  2024-02-20   Late but recorded
(Most of 2023 was effectively continuous Stage 4+, so we add overlap-prevention.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


EVENTS = [
    "2019-02-11",
    "2019-12-09",
    "2022-06-21",
    "2022-09-18",
    "2022-12-01",
    "2023-01-15",
    "2023-09-15",
    "2024-02-20",
]


def main():
    try:
        pl = load_prices(["PL=F"], start="2015-01-01").iloc[:, 0].rename("PL")
    except Exception as e:
        return mark_failed("T8_eskom_load_shedding", f"price load: {e}")

    if pl.empty:
        return mark_failed("T8_eskom_load_shedding", "PL=F empty")

    pl_r = pl.pct_change()

    HOLD = 42  # 2 months
    pos = pd.Series(0.0, index=pl.index)
    starts = []
    for d in EVENTS:
        d_ts = pd.Timestamp(d)
        i = pl.index.searchsorted(d_ts) + 1
        if i >= len(pl.index):
            continue
        starts.append(str(pl.index[i].date()))
        end = min(i + HOLD, len(pl.index))
        pos.iloc[i:end] = 1.0

    pnl = (pos.shift(1) * pl_r).dropna()
    bench = pl_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T8 Eskom Stage >=4 -> long PL=F 2mo")
    print_metrics(m)
    print(f"events: {len(EVENTS)}, in-position days: {int((pos > 0).sum())}")

    save_result("T8_eskom_load_shedding", m, extra={
        "status": "ok",
        "rule": "Stage 4+ load shedding for 7+ days -> long PL=F 2 months",
        "mechanism": ("South Africa supplies ~70% of mined platinum; sustained "
                      "Stage 4+ shedding curtails Sibanye/Anglo/Impala "
                      "production -> supply shock -> Pt price up."),
        "source": "Curated event list from Wikipedia/Eskom/Daily Maverick reporting",
        "n_events": len(EVENTS),
        "event_dates": EVENTS,
        "caveats": ("Curated events (not API-driven) - sample is small, "
                    "and 2023 was effectively continuous Stage 4+, which we "
                    "represented as two distinct events (Jan & Sep)."),
    })


if __name__ == "__main__":
    main()
