"""
N8 EIA propane stocks autumn squeeze.
Weekly EIA propane+propylene stocks (WPRSTUS1). When stocks during Sep 1 - Nov 15 fall
below 5-year same-week minimum, long UNG for 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import io
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_propane():
    fp = DATA / "eia_propane_stocks.parquet"
    if fp.exists():
        return pd.read_parquet(fp).iloc[:, 0]
    url = "https://www.eia.gov/dnav/pet/hist_xls/WPRSTUS1w.xls"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    xl = pd.ExcelFile(io.BytesIO(raw))
    df = xl.parse("Data 1", header=2)
    df.columns = ["Date", "Stocks"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df["Stocks"] = pd.to_numeric(df["Stocks"], errors="coerce")
    df = df.dropna()
    df.to_parquet(fp)
    return df.iloc[:, 0]


def main():
    try:
        propane = fetch_propane()
        ung = load_prices(["UNG"], start="1995-01-01").iloc[:, 0].rename("UNG")
    except Exception as e:
        return mark_failed("N8_eia_propane", f"data load failed: {e}")

    # week-of-year for same-week comparison
    prop = propane.copy()
    prop.index = pd.to_datetime(prop.index)
    df = pd.DataFrame({"stocks": prop})
    df["woy"] = df.index.isocalendar().week.astype(int)
    df["year"] = df.index.year

    # For each weekly observation in Sep 1 - Nov 15 window, compute 5-year same-week min
    triggers = []
    for d, row in df.iterrows():
        if not (pd.Timestamp(d.year, 9, 1) <= d <= pd.Timestamp(d.year, 11, 15)):
            continue
        window_start = pd.Timestamp(d.year - 5, 1, 1)
        prior = df[(df.index >= window_start) & (df.index < pd.Timestamp(d.year, 1, 1))]
        same_week = prior[prior["woy"] == row["woy"]]
        if len(same_week) < 3:
            continue
        if row["stocks"] < same_week["stocks"].min():
            triggers.append(d)

    ung_rets = ung.pct_change()
    pos = pd.Series(0.0, index=ung_rets.index)
    for d in triggers:
        loc = ung_rets.index.searchsorted(d)
        # apply for next 30 trading days
        for k in range(1, 31):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if not triggers:
        return mark_failed("N8_eia_propane", "no qualifying triggers (no autumn week below 5y min)",
                           extra={"n_weeks_checked": int((df.index.month.isin([9,10,11])).sum())})

    pnl_full = (pos * ung_rets).dropna()
    pnl = pnl_full.loc[pnl_full.ne(0).cummax()]
    bench = ung_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N8 EIA propane autumn -> UNG")
    m["n_triggers"] = len(triggers)
    print(f"Triggers: {len(triggers)}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("N8_eia_propane", m, extra={
        "status": "ok",
        "rule": "When EIA propane stocks (WPRSTUS1) during Sep 1 - Nov 15 fall below 5-year same-week minimum, long UNG 30 sessions.",
        "mechanism": "Pre-heating-season propane draw signals tight balance -> natgas substitution / cold-trade premium",
        "universe": "UNG",
        "source": "EIA Weekly Petroleum Status Report (Data 1 sheet of WPRSTUS1w.xls)",
        "n_triggers": len(triggers),
    })


if __name__ == "__main__":
    main()
