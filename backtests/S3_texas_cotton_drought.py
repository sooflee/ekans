"""
S3 Texas cotton drought -> long CT=F.

Rule: Compute weekly U.S. Drought Monitor (USDM) D2-or-worse area share of TX.
When share > 50% during May-July, long CT=F (cotton front-month) for ~63
trading days (3 months). Non-overlapping events.

Mechanism: Texas High Plains supply ~40% of U.S. cotton; D2+ drought in
spring/early summer impairs planting, emergence and pod-set, shifting USDA
production forecasts down and bidding up nearby cotton futures.

Source: U.S. Drought Monitor Data Services API (StateStatistics, aoi=48 for TX,
statisticsType=1 i.e. percent of area). yfinance CT=F.

Note: BAL ETF cited in spec is delisted on Yahoo; using CT=F directly.
"""
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA


def fetch_usdm_tx():
    cache = DATA / "usdm_tx_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    base = "https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent"
    params = {
        "aoi": "48",  # Texas FIPS
        "startdate": "1/1/2000",
        "enddate": "12/31/2026",
        "statisticsType": "1",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    df = pd.DataFrame(data)
    df["mapDate"] = pd.to_datetime(df["mapDate"])
    for c in ["d0", "d1", "d2", "d3", "d4", "none"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("mapDate").set_index("mapDate")
    df.to_parquet(cache)
    return df


def main():
    try:
        usdm = fetch_usdm_tx()
    except Exception as e:
        return mark_failed("S3_texas_cotton_drought", f"USDM fetch failed: {e}")
    if len(usdm) < 200:
        return mark_failed("S3_texas_cotton_drought", f"insufficient USDM data ({len(usdm)} obs)")

    # d2 column = % area in D2+ severity (already cumulative in USDM definition)
    d2_plus = usdm["d2"].dropna()

    try:
        ct = load_prices(["CT=F"], start="2000-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S3_texas_cotton_drought", f"CT=F load failed: {e}")

    rets = ct.pct_change()

    # Trigger: D2+ > 50% during May-July
    mask = (d2_plus > 50) & (d2_plus.index.month.isin([5, 6, 7]))
    triggers = d2_plus.index[mask].tolist()
    print(f"USDM TX obs: {len(usdm)}; D2+ max: {d2_plus.max():.1f}%")
    print(f"May-Jul weeks with D2+ > 50: {len(triggers)}")

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    HOLD = 63  # ~3 months
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S3_texas_cotton_drought",
                           f"no qualifying triggers (max D2+ = {d2_plus.max():.1f}%)")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S3 TX D2+>50% May-Jul -> long CT=F 63d")
    m["n_events"] = n_events
    print(f"Non-overlap events: {n_events}; first: {event_dates[:5]}")
    print_metrics(m)

    save_result("S3_texas_cotton_drought", m, extra={
        "status": "ok",
        "rule": "When USDM Texas D2+ severity area-share > 50% during May-July, long CT=F next session for 63 trading days; non-overlapping events.",
        "mechanism": "Texas High Plains is the dominant U.S. cotton supplier; spring/early-summer severe drought (D2+) impairs planting and pod-set, depressing forecast production and lifting nearby cotton futures.",
        "source": "U.S. Drought Monitor Data Services API (StateStatistics, aoi=48); yfinance CT=F. NOTE: spec's BAL ETF (iPath cotton) delisted on Yahoo; using CT=F directly. State-level D2+ used as proxy for Texas High Plains cotton-belt drought (county-level shapefiles parse not implemented in this run).",
        "n_events": n_events,
        "first_events": event_dates[:5],
    })


if __name__ == "__main__":
    main()
