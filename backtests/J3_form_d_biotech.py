"""
J3 SEC Form D biotech filings surge -> short XBI.

Data: SEC DERA Form D structured data quarterly zips, e.g.
   https://www.sec.gov/files/structureddata/data/form-d-data-sets/2024q1_d.zip

We classify biotech via OFFERING.tsv's INDUSTRYGROUPTYPE in
{'Biotechnology', 'Pharmaceuticals'}. The SIC_CODE column in
FORMDSUBMISSION is empty for most recent filings, so industry group
is the working substitute.

Rule:
- Weekly count of biotech Form D filings (using FILING_DATE).
- For each week, compute trailing-52-week 90th percentile of the
  weekly counts. When current count > that 90th-pct threshold,
  go short XBI for next 60 trading days. (Dedupe overlapping triggers.)
"""
import sys
import io
import re
import time
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)

UA = {"User-Agent": "ekans research benson@example.com"}
BIOTECH_GROUPS = {"Biotechnology", "Pharmaceuticals"}


def fetch_quarter(year, quarter):
    cache = DATA / f"formd_biotech_{year}q{quarter}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = f"https://www.sec.gov/files/structureddata/data/form-d-data-sets/{year}q{quarter}_d.zip"
    r = requests.get(url, timeout=120, headers=UA)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # find subdir
    names = z.namelist()
    sub = None
    for n in names:
        if n.upper().endswith("/OFFERING.TSV") or n.upper().endswith("OFFERING.TSV"):
            sub = n
            break
    if sub is None:
        return None
    with z.open(sub) as f:
        off = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    # find submission file in same dir
    base = sub.rsplit("/", 1)[0] if "/" in sub else ""
    sub_path = f"{base}/FORMDSUBMISSION.tsv" if base else "FORMDSUBMISSION.tsv"
    if sub_path not in names:
        # try matching case-insensitively
        for n in names:
            if n.upper().endswith("FORMDSUBMISSION.TSV"):
                sub_path = n
                break
    with z.open(sub_path) as f:
        sub_df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    df = off[["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE"]].merge(
        sub_df[["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE"]],
        on="ACCESSIONNUMBER", how="inner",
    )
    df["FILING_DATE"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["FILING_DATE"])
    biotech = df[df["INDUSTRYGROUPTYPE"].isin(BIOTECH_GROUPS)].copy()
    biotech = biotech[["FILING_DATE", "INDUSTRYGROUPTYPE", "ACCESSIONNUMBER"]]
    biotech.to_parquet(cache)
    return biotech


def build_biotech_filings():
    cache = DATA / "formd_biotech_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    # 2014-present (SEC DERA Form D dataset starts 2008Q1 but biotech tagging
    # consistent from ~2014; we cap at present quarter).
    today = pd.Timestamp.today()
    cur_q = (today.month - 1) // 3 + 1
    cur_y = today.year
    for y in range(2014, cur_y + 1):
        for q in range(1, 5):
            if y == cur_y and q > cur_q:
                continue
            try:
                df = fetch_quarter(y, q)
                if df is not None and not df.empty:
                    parts.append(df)
                    print(f"  {y}q{q}: {len(df)} biotech filings")
                time.sleep(0.3)
            except Exception as e:
                print(f"  {y}q{q} err: {e}")
    if not parts:
        raise RuntimeError("No Form D data fetched")
    all_df = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSIONNUMBER")
    all_df.to_parquet(cache)
    return all_df


def main():
    try:
        df = build_biotech_filings()
    except Exception as e:
        return mark_failed("J3_form_d_biotech", f"SEC Form D fetch failed: {e}")
    if df.empty:
        return mark_failed("J3_form_d_biotech", "No biotech Form D filings")

    df["FILING_DATE"] = pd.to_datetime(df["FILING_DATE"])
    # weekly count (week ending Friday)
    weekly = df.set_index("FILING_DATE").resample("W-FRI").size().rename("count")
    # trailing 52-week 90th percentile
    p90 = weekly.rolling(52, min_periods=20).quantile(0.90)
    triggers = weekly[weekly > p90].index

    px = load_prices(["XBI", "SPY"], start="2014-01-01")
    if px.empty or "XBI" not in px.columns:
        return mark_failed("J3_form_d_biotech", "XBI load failed")
    rets = px.pct_change()

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
        end_idx = min(idx + 60, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    pnl = (daily_pos.shift(1) * rets["XBI"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J3 Biotech Form D surge -> short XBI")
    print_metrics(m)
    print(f"\nWeeks: {len(weekly)} ; trigger weeks: {len(triggers)} ; dedup events: {n_events}")

    save_result("J3_form_d_biotech", m, extra={
        "status": "ok",
        "rule": ("Weekly biotech Form D filing count > trailing 52-week 90th percentile -> "
                 "short XBI for 60 trading days (dedup overlapping triggers)."),
        "mechanism": "Cluster of private biotech offerings signals supply-side overheating and overhanging dilution risk for public biotech.",
        "source": "https://www.sec.gov/dera/data/form-d (quarterly structured Form D zips, OFFERING.INDUSTRYGROUPTYPE in {Biotechnology, Pharmaceuticals}).",
        "n_events": n_events,
        "n_weeks": int(len(weekly)),
        "caveats": "Used INDUSTRYGROUPTYPE (not SIC); SIC_CODE field is largely blank in modern Form D submissions.",
    })


if __name__ == "__main__":
    main()
