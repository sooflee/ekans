"""
F02 NAAIM exposure
Fetch NAAIM exposure index xlsx from naaim.org (link discovered from HTML page).
Rule: 4-week MA NAAIM < 30 -> long SPY 8 weeks; > 90 -> flat.
"""
import sys
import io
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def find_naaim_xlsx():
    import requests
    page = "https://www.naaim.org/programs/naaim-exposure-index/"
    r = requests.get(page, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} on landing page")
    links = re.findall(r'href="([^"]*\.(?:xlsx|xls|csv))"', r.text)
    if not links:
        raise RuntimeError("no data link found on NAAIM page")
    return links[0]


def main():
    try:
        import requests
        data_url = find_naaim_xlsx()
        r = requests.get(data_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} fetching {data_url}")
        df = pd.read_excel(io.BytesIO(r.content), header=None)
    except Exception as e:
        return mark_failed("F02_naaim_exposure",
                          f"NAAIM data not cleanly fetchable; cite NAAIM site directly. ({e})")

    # Row 0 contains headers in xlsx
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    cols = [str(c) for c in df.columns]
    date_col = next((c for c in cols if "date" in c.lower()), None)
    mean_col = next((c for c in cols if "mean" in c.lower() or "average" in c.lower()), None)
    if not (date_col and mean_col):
        return mark_failed("F02_naaim_exposure",
                          f"NAAIM columns unrecognized: {cols[:10]}")

    df = df[[date_col, mean_col]].copy()
    df.columns = ["Date", "NAAIM"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["NAAIM"] = pd.to_numeric(df["NAAIM"], errors="coerce")
    df = df.dropna().sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="last").set_index("Date")

    ma4 = df["NAAIM"].rolling(4).mean()
    long_trig = ma4 < 30
    flat_trig = ma4 > 90

    spy = load_prices(["SPY"], start="2006-01-01").iloc[:, 0].rename("SPY")
    rets = spy.pct_change()

    weekly_pos = pd.Series(0.0, index=ma4.index)
    rem_long = 0
    rem_flat = 0
    for i in range(len(ma4)):
        if long_trig.iloc[i]:
            rem_long = 8
        if flat_trig.iloc[i]:
            rem_flat = 8
        if rem_flat > 0:
            weekly_pos.iloc[i] = 0.0
            rem_flat -= 1
            if rem_long > 0:
                rem_long -= 1
        elif rem_long > 0:
            weekly_pos.iloc[i] = 1.0
            rem_long -= 1

    daily_pos = weekly_pos.reindex(spy.index, method="ffill").fillna(0.0)
    pnl = (daily_pos.shift(1) * rets).dropna()

    m = compute_metrics(pnl, benchmark=rets, name="F02 NAAIM exposure")
    print_metrics(m)
    save_result("F02_naaim_exposure", m, extra={
        "status": "ok",
        "rule": "4w MA NAAIM <30 -> long SPY 8w; >90 -> flat 8w.",
        "data_source": data_url,
    })


if __name__ == "__main__":
    main()
