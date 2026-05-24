"""
J8 USGS copper force-majeure events -> long FCX.

USGS Mineral Industry Surveys are PDFs and not amenable to a structured
free pull. Instead we hardcode known supply-disruption events in copper
since 2015 and test a long FCX (Freeport, COPX is short history) holding
of 30 trading days starting from the event date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


# Major copper supply disruptions (start dates). All sourced from public news.
EVENTS = [
    ("2017-02-09", "Escondida BHP strike begins"),       # ~43 days
    ("2017-04-01", "Grasberg Freeport export halt"),
    ("2018-08-01", "Escondida wage talks deadlock"),
    ("2019-12-09", "Las Bambas roadblock by Cotabambas communities"),
    ("2020-04-13", "Antofagasta covid disruption"),
    ("2021-11-15", "Las Bambas protests halt operations"),
    ("2022-04-20", "Las Bambas 50-day blockade begins"),
    ("2022-12-14", "Cuajone Peru mine seized by protesters"),
    ("2023-11-28", "Cobre Panama First Quantum operations suspended"),
    ("2024-04-22", "Antamina pit-wall incident halt"),
]


def main():
    px = load_prices(["FCX", "SPY"], start="2015-01-01")
    if px.empty or "FCX" not in px.columns:
        return mark_failed("J8_usgs_copper", "FCX load failed")
    rets = px.pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d_str, desc in EVENTS:
        d = pd.Timestamp(d_str)
        nxt = rets.index[rets.index >= d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = 1.0
        n_events += 1

    pnl = (daily_pos.shift(1) * rets["FCX"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J8 Copper supply shocks -> long FCX")
    print_metrics(m)
    print(f"\nEvents: {n_events}")

    save_result("J8_usgs_copper", m, extra={
        "status": "ok",
        "rule": "On each hand-curated copper supply-disruption event date, long FCX for 30 trading days.",
        "mechanism": "Mine-level supply disruptions tighten copper market and lift producer equities (FCX).",
        "source": "Hand-curated from public press coverage (Escondida, Las Bambas, Cobre Panama, Grasberg, Antamina, Antofagasta).",
        "n_events": n_events,
        "events": [{"date": d, "description": desc} for d, desc in EVENTS],
        "caveats": "Small-sample hardcoded events; selection-on-known-outcome bias; USGS PDFs not parsed.",
    })


if __name__ == "__main__":
    main()
