"""
J4 Mississippi River low water at Memphis -> long CORN.

Spec wants stage from NOAA AHPS but water.weather.gov is unreachable from this env.
Substitute USGS discharge at Memphis (site 07032000, parameterCd=00060 cubic-ft/sec),
the same hydrologic measurement: when river is low, discharge is low.

We define "low" as discharge below the 5th percentile of daily discharge in the
training base (1990-2010), restricted to Jul-Oct (low-water season).
When 5 consecutive days below that low threshold during Jul-Oct, go long CORN ETF
for 30 trading days. Compare to CORN buy-and-hold.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)


def fetch_usgs_memphis(start="1990-01-01", end="2024-12-31"):
    cache = DATA / "usgs_memphis_07032000.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites=07032000"
           f"&startDT={start}&endDT={end}&parameterCd=00060&statCd=00003")
    r = requests.get(url, timeout=120)
    j = json.loads(r.text)
    ts = j.get("value", {}).get("timeSeries", [])
    if not ts:
        raise RuntimeError("USGS Memphis returned no series")
    vals = ts[0]["values"][0]["value"]
    df = pd.DataFrame(vals)
    df["dateTime"] = pd.to_datetime(df["dateTime"]).dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("dateTime")[["value"]].rename(columns={"value": "discharge_cfs"})
    df = df.sort_index()
    df.to_parquet(cache)
    return df


def main():
    try:
        df = fetch_usgs_memphis()
    except Exception as e:
        return mark_failed("J4_mississippi_river",
                           f"USGS Memphis fetch failed: {e}")
    if df.empty:
        return mark_failed("J4_mississippi_river", "Empty USGS data")

    discharge = df["discharge_cfs"].dropna()
    discharge.index = pd.DatetimeIndex(discharge.index)

    # threshold = 5th percentile of Jul-Oct discharge in 1990-2010 base period
    base = discharge[(discharge.index.year <= 2010) & (discharge.index.month.isin([7,8,9,10]))]
    thresh = base.quantile(0.05)

    # consecutive-5-days-below-thresh during Jul-Oct
    in_season = discharge.index.month.isin([7,8,9,10])
    below = (discharge < thresh) & in_season
    # rolling sum of "below" over 5 days
    consec5 = below.rolling(5).sum()
    triggers = consec5[consec5 == 5].index

    # collapse triggers: only fire if we are not already in a 30d holding window
    px = load_prices(["CORN"], start="2010-06-01")
    if px.empty:
        return mark_failed("J4_mississippi_river", "CORN load failed")
    corn = px["CORN"].dropna()
    rets = corn.pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    pnl = (daily_pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J4 Mississippi low water -> long CORN")
    print_metrics(m)
    print(f"\nTrigger events (deduplicated): {n_events}")

    save_result("J4_mississippi_river", m, extra={
        "status": "ok",
        "rule": ("USGS discharge at Memphis (site 07032000) < 5th-pct of Jul-Oct 1990-2010 base "
                 "for 5 consecutive days during Jul-Oct -> long CORN ETF for 30 trading days."),
        "mechanism": "Low Mississippi water raises grain barge freight costs and disrupts export logistics, supporting domestic grain prices.",
        "source": "https://waterservices.usgs.gov/nwis/dv/?sites=07032000 (proxy for NOAA AHPS stage which is unreachable from this env)",
        "n_events": n_events,
        "caveats": "Spec asked for stage (feet); we use discharge (cfs) at the same Memphis site as a proxy. CORN ETF only trades since 2010, so sample is small.",
    })


if __name__ == "__main__":
    main()
