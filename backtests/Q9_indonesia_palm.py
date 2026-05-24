"""
Q9 Indonesia palm oil policy events -> proxy via ZL=F (CBOT soybean oil).

Indonesia is the world's largest palm oil exporter; policy shocks (export ban, DMO,
levy hikes) historically spike global vegetable-oil prices. Malaysian FCPO is not on
yfinance; we use ZL=F (CBOT soybean oil futures) as a substitute - imperfect but
liquid and historically highly correlated with palm during shocks.

Hardcoded major events (Indonesia palm oil policy news):
- 2022-01-27: Indonesia DMO (Domestic Market Obligation) announced
- 2022-04-22: Total palm export ban announced (effective Apr 28)
- 2022-05-23: Ban lifted; export levy reinstated
- 2022-09-09: DMO/levy update (export multiplier raised)
- 2023-02-08: Indonesia tightens export permits (DMO ratio hiked)
- 2024-01-05: Biodiesel B35 raised - cuts exports
- 2024-08-12: Indonesia mulls B40 mandate (cuts exports further)

Rule: On each event date, long ZL=F next session for 20 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)

EVENTS = [
    ("2022-01-27", "DMO announced"),
    ("2022-04-22", "Total export ban announced"),
    ("2022-05-23", "Export ban lifted"),
    ("2022-09-09", "DMO/levy update"),
    ("2023-02-08", "Export permit tightening"),
    ("2024-01-05", "B35 biodiesel mandate"),
    ("2024-08-12", "B40 mandate signals"),
]


def main():
    try:
        zl = load_prices(["ZL=F"], start="2020-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("Q9_indonesia_palm", f"ZL=F load failed: {e}")
    rets = zl.pct_change()

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    detail = []
    for ds, label in EVENTS:
        d = pd.Timestamp(ds)
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 20, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        detail.append({"event": ds, "label": label, "entry": str(start.date())})

    if n_events == 0:
        return mark_failed("Q9_indonesia_palm", "no qualifying event dates fell in trading range")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q9 Indonesia palm events -> long ZL=F 20d")
    m["n_events"] = n_events
    print(f"Events: {n_events}")
    for d in detail:
        print(f"  {d}")
    print_metrics(m)

    save_result("Q9_indonesia_palm", m, extra={
        "status": "ok",
        "rule": "On each hardcoded Indonesia palm-oil policy event date, long ZL=F (CBOT soybean oil, palm proxy) next session for 20 trading days; non-overlapping.",
        "mechanism": "Indonesian export curbs / DMO / biodiesel mandates restrict global veg-oil supply -> SBO and palm rally together via substitution.",
        "source": "Hardcoded event dates from Reuters/Bloomberg public news; yfinance ZL=F as palm proxy (FCPO unavailable on yfinance).",
        "n_events": n_events,
        "events": detail,
        "notes": "ZL=F is an imperfect proxy for palm; results indicative only.",
    })


if __name__ == "__main__":
    main()
