"""
M9 Refinery utilization shock.
EIA weekly US refinery operable-capacity utilization (PET.WPULEUS3.W).
When weekly utilization drops >4 percentage points WoW, long XLE for 30 days.
Mechanism: sharp refinery shutdowns -> crack spreads widen -> XLE pops on integrated/refiner mix.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
    DATA,
)


EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/WPULEUS3w.xls"
CACHE = DATA / "eia_WPULEUS3.parquet"


def fetch_util():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    req = urllib.request.Request(EIA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    df = pd.read_excel(io.BytesIO(data), sheet_name="Data 1", skiprows=2)
    df.columns = ["Date", "util"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna().set_index("Date").sort_index()
    df.to_parquet(CACHE)
    return df


def main():
    try:
        util = fetch_util()["util"]
        xle = load_prices(["XLE"], start="1999-01-01").iloc[:, 0].rename("XLE")
    except Exception as e:
        return mark_failed("M9_refinery_utilization", f"data load failed: {e}")

    wow = util.diff()  # percentage point change
    trigger = (wow < -4.0)
    # First-cross
    prev = trigger.shift(1).fillna(False)
    first_cross = trigger & (~prev)
    event_dates = util.index[first_cross]

    rets = xle.pct_change()
    pos = pd.Series(0.0, index=xle.index)
    n_events = 0
    for d in event_dates:
        ix = xle.index.searchsorted(d)
        if ix >= len(xle.index):
            continue
        end_ix = min(ix + 30, len(xle.index))
        pos.iloc[ix:end_ix] = 1.0
        n_events += 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M9 Refinery util drop >4pp WoW → long XLE 30d")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M9_refinery_utilization", m, extra={
        "status": "ok",
        "rule": "When weekly US refinery utilization drops >4 percentage points WoW, long XLE for 30 trading days.",
        "mechanism": "Sharp refinery outages (hurricanes, fires, planned strike turnarounds) widen crack spreads, lifting integrated energy.",
        "source": "EIA WPULEUS3 weekly series (XLS).",
    })


if __name__ == "__main__":
    main()
