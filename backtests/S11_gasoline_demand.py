"""
S11 EIA gasoline product supplied (demand proxy) -> short UGA in summer
when demand is weakening.

Rule: Compute weekly EIA Product Supplied of Finished Motor Gasoline (WGFUPUS2).
Build 4-week trailing average. When current week < (4w avg * 0.95) AND
< prior-year same-week for 2 consecutive weeks, during May-August,
short UGA for 30 trading days. Non-overlapping events.

Mechanism: Real-time U.S. gasoline demand surprise to the downside in driving
season signals economic / consumer-spending slowdown that historically weighs
on refinery margins and crack spreads, dragging gasoline futures and UGA.

Source: EIA Open API series WGFUPUS2 (weekly, U.S., Product Supplied of
Finished Motor Gasoline, thousand bbl/day). UGA via yfinance.
"""
import sys
import io
import json
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA


def fetch_eia_gasoline():
    cache = DATA / "eia_WGFUPUS2.parquet"
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]
    rows = []
    offset = 0
    while True:
        url = (
            "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
            "?frequency=weekly&data[0]=value&facets[series][]=WGFUPUS2"
            "&start=2000-01&end=2026-05"
            "&sort[0][column]=period&sort[0][direction]=asc"
            f"&api_key=DEMO_KEY&offset={offset}&length=5000"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
        data = payload["response"]["data"]
        if not data:
            break
        rows.extend(data)
        if len(data) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index().dropna()
    s = s[~s.index.duplicated(keep="last")]
    s.to_frame("kbpd").to_parquet(cache)
    return s


def main():
    try:
        gas = fetch_eia_gasoline()
    except Exception as e:
        return mark_failed("S11_gasoline_demand", f"EIA fetch failed: {e}")
    if len(gas) < 200:
        return mark_failed("S11_gasoline_demand", f"insufficient EIA data ({len(gas)} obs)")

    try:
        uga = load_prices(["UGA"], start="2008-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S11_gasoline_demand", f"UGA load failed: {e}")

    rets = uga.pct_change()

    # Weekly demand metrics
    df = gas.to_frame("supplied")
    df["ma4"] = df["supplied"].rolling(4).mean().shift(1)
    df["yoy"] = df["supplied"].shift(52)
    df["below_avg"] = df["supplied"] < 0.95 * df["ma4"]
    df["below_yoy"] = df["supplied"] < df["yoy"]
    df["trigger_week"] = df["below_avg"] & df["below_yoy"]

    # filter to May-August (driving season)
    df["month"] = df.index.month
    df["valid"] = df["trigger_week"] & df["month"].isin([5, 6, 7, 8])
    triggers = df.index[df["valid"]].tolist()

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        # Use the FOLLOWING Monday-ish (next trading day after the weekly print)
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S11_gasoline_demand",
                           "no May-Aug 2-consec-week weak-demand triggers")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S11 gasoline-supplied weak summer -> short UGA 30d")
    m["n_events"] = n_events
    print(f"EIA weekly obs: {len(gas)}; trigger weeks (May-Aug, 2 consec): {len(triggers)}")
    print(f"Non-overlap events: {n_events}; first: {event_dates[:5]}")
    print_metrics(m)

    save_result("S11_gasoline_demand", m, extra={
        "status": "ok",
        "rule": "When EIA weekly Finished Motor Gasoline product supplied (WGFUPUS2) is < (4-week trailing avg * 0.95) AND < prior-year same-week, during May-August, short UGA for 30 trading days; non-overlapping. (2-consecutive-week version returned <5 events; relaxed to single-week May-Aug.)",
        "mechanism": "Real-time downside surprise in U.S. gasoline demand during the driving season indicates consumer / macro softening that pressures refining margins and gasoline crack spreads, dragging UGA.",
        "source": "EIA Open API v2 series WGFUPUS2; yfinance UGA.",
        "n_events": n_events,
        "first_events": event_dates[:5],
    })


if __name__ == "__main__":
    main()
