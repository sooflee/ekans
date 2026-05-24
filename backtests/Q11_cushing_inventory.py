"""
Q11 Cushing crude inventory - low-stocks signal.

EIA Weekly Cushing crude stocks series WCESTUS1 (Weekly Cushing OK Ending Stocks).
Via FRED (mnemonic WCESTUS1) if available, else EIA DNAV XLS.

Rule: When weekly Cushing stocks fall below 25,000 (thousand barrels = 25 MMbbl),
go long CL=F front-month for 30 trading days. Non-overlapping.
"""
import sys
import io
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed, DATA
)


def fetch_cushing():
    cache = DATA / "eia_cushing_wcestus1.parquet"
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]
    # Try FRED first
    try:
        df = load_fred(["WCESTUS1"], start="1990-01-01")
        s = df.iloc[:, 0].dropna()
        if len(s) > 100:
            s.to_frame("stocks").to_parquet(cache)
            return s
    except Exception:
        pass
    # EIA DNAV XLS fallback
    url = "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw.xls"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    xl = pd.ExcelFile(io.BytesIO(raw))
    # Find Data sheet
    df = None
    for n in xl.sheet_names:
        if n.startswith("Data"):
            sub = pd.read_excel(io.BytesIO(raw), sheet_name=n, skiprows=2)
            if sub.shape[1] >= 2:
                df = sub
                break
    if df is None:
        raise RuntimeError("could not parse Cushing XLS")
    df.columns = ["date", "stocks"] + list(df.columns[2:])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")["stocks"]
    df = pd.to_numeric(df, errors="coerce").dropna().sort_index()
    df.to_frame("stocks").to_parquet(cache)
    return df


def main():
    try:
        stocks = fetch_cushing()
    except Exception as e:
        return mark_failed("Q11_cushing_inventory", f"Cushing data fetch failed: {e}")
    if len(stocks) < 100:
        return mark_failed("Q11_cushing_inventory", f"insufficient Cushing data ({len(stocks)} obs)")

    try:
        cl = load_prices(["CL=F"], start="2000-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("Q11_cushing_inventory", f"CL=F load failed: {e}")

    rets = cl.pct_change()

    # Threshold: < 25,000 thousand bbl = 25 MMbbl
    triggers = stocks[stocks < 25_000].index

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in triggers:
        idx = rets.index.searchsorted(d)
        if idx >= len(rets.index):
            continue
        start = rets.index[idx]
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    if n_events == 0:
        return mark_failed("Q11_cushing_inventory",
                           f"no qualifying triggers (min Cushing seen={stocks.min():.0f} kbbl)")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q11 Cushing<25MMbbl -> long CL=F 30d")
    m["n_events"] = n_events
    print(f"Cushing obs: {len(stocks)}; triggers: {len(triggers)}; events: {n_events}")
    print(f"Cushing min: {stocks.min():.0f} kbbl ; threshold 25,000 kbbl")
    print_metrics(m)

    save_result("Q11_cushing_inventory", m, extra={
        "status": "ok",
        "rule": "When weekly EIA Cushing (OK) crude stocks (WCESTUS1) < 25,000 thousand bbl, long CL=F for 30 trading days; non-overlapping events.",
        "mechanism": "Low Cushing stocks signal tight WTI delivery hub, supporting front-month WTI; historically followed by backwardation and price strength.",
        "source": "FRED WCESTUS1 / EIA Weekly Cushing Crude Oil Stocks (W_EPC0_SAX_YCUOK_MBBL)",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
