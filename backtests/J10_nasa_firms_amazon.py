"""
J10 NASA FIRMS Amazon fire spikes -> long SOYB.

NASA FIRMS API needs a registration token. Instead use hardcoded years
where INPE / press reported major Amazon fire seasons:
2019 (peak Aug-Sep), 2020, 2023.
For each, long SOYB ETF for 60 trading days starting Sept 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


FIRE_YEARS = [2019, 2020, 2023]


def main():
    px = load_prices(["SOYB", "SPY"], start="2011-09-01")
    if px.empty or "SOYB" not in px.columns:
        return mark_failed("J10_nasa_firms_amazon", "SOYB load failed")
    rets = px.pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    for y in FIRE_YEARS:
        start_target = pd.Timestamp(f"{y}-09-01")
        nxt = rets.index[rets.index >= start_target]
        if len(nxt) == 0:
            continue
        idx = rets.index.get_loc(nxt[0])
        end_idx = min(idx + 60, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = 1.0
        n_events += 1

    pnl = (daily_pos.shift(1) * rets["SOYB"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J10 Amazon fires -> long SOYB")
    print_metrics(m)
    print(f"\nEvents: {n_events}")

    save_result("J10_nasa_firms_amazon", m, extra={
        "status": "ok",
        "rule": "For each hand-curated Amazon fire-spike year (2019/2020/2023), long SOYB for 60 trading days from Sept 1.",
        "mechanism": "Brazilian deforestation fires tied to crop expansion and dry-season risk; severe fire seasons can disrupt soy logistics and lift soy prices.",
        "source": "Hand-curated from INPE/press coverage; NASA FIRMS API at https://firms.modaps.eosdis.nasa.gov requires registration token (skipped).",
        "n_events": n_events,
        "events": FIRE_YEARS,
        "caveats": "Very small n; selection-on-known-outcome bias; FIRMS API not parsed.",
    })


if __name__ == "__main__":
    main()
