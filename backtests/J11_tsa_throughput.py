"""
J11 TSA throughput weak -> short JETS.

TSA publishes a daily table at https://www.tsa.gov/travel/passenger-volumes.
The page shows last ~1 year. We supplement with web.archive.org snapshots
(2022 snapshot has 2019-2022; 2024 snapshot has 2020-2024) to assemble
2019-present daily throughput.

Rule: 7-day avg of TSA throughput < 95% of same-day 2019 baseline on a
weekday -> short JETS for 30 trading days, NOT compounding (one position
at a time; deduplicate overlapping triggers).
"""
import sys
import io
import re
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)

UA = {"User-Agent": "Mozilla/5.0 (compatible; ekans-research/1.0)"}


def parse_tsa_table(html, year_cols_header_index=None):
    """Parse the TSA throughput HTML table.

    Returns DataFrame with index=date, single column 'throughput'.
    The table has columns like: Date | 2024 | 2023 | 2022 | 2021 | 2020 | 2019
    The first numeric column is the most recent year (matching the row's date),
    and subsequent columns are same-month-day in prior years.
    """
    rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
    out = {}
    header_years = None
    for row in rows:
        # try header detection
        ths = re.findall(r'<th[^>]*>([^<]*)</th>', row)
        if ths and not header_years:
            cleaned = [t.strip() for t in ths]
            # look for year-like cells
            years = [int(c) for c in cleaned if c.strip().isdigit() and 2017 <= int(c.strip()) <= 2030]
            if years:
                header_years = years
            continue
        cells = re.findall(r'<td[^>]*>([^<]*)</td>', row)
        cells = [c.strip() for c in cells]
        if not cells:
            continue
        # first cell is date string like '12/25/2021'
        m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', cells[0])
        if not m:
            continue
        mo, dy, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # subsequent cells are numbers (with commas), one per year column
        nums = []
        for c in cells[1:]:
            c = c.replace(",", "").strip()
            if c == "":
                nums.append(None)
            else:
                try:
                    nums.append(int(c))
                except ValueError:
                    nums.append(None)
        # Determine year-per-column. If header_years known, use them; otherwise
        # the first numeric column corresponds to the year `yr` from the date,
        # and prior columns are yr-1, yr-2, ...
        if header_years and len(header_years) == len(nums):
            for y, v in zip(header_years, nums):
                if v is None:
                    continue
                try:
                    out[pd.Timestamp(year=y, month=mo, day=dy)] = v
                except ValueError:
                    pass
        else:
            cur_year = yr
            for i, v in enumerate(nums):
                if v is None:
                    continue
                y = cur_year - i
                try:
                    out[pd.Timestamp(year=y, month=mo, day=dy)] = v
                except ValueError:
                    pass
    if not out:
        return pd.DataFrame(columns=["throughput"])
    s = pd.Series(out, name="throughput").sort_index()
    return s.to_frame()


def fetch_tsa_history():
    cache = DATA / "tsa_throughput.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    snapshots = [
        # late-2024 wayback snapshot: gives 2019-2024 partial
        "https://web.archive.org/web/20241231122332/https://www.tsa.gov/travel/passenger-volumes",
        # late-2022 snapshot
        "https://web.archive.org/web/20221222161752/https://www.tsa.gov/travel/passenger-volumes",
        # current live page
        "https://www.tsa.gov/travel/passenger-volumes",
    ]
    all_df = []
    for u in snapshots:
        try:
            r = requests.get(u, timeout=90, headers=UA)
            if r.status_code != 200:
                continue
            df = parse_tsa_table(r.text)
            if not df.empty:
                all_df.append(df)
            time.sleep(1)
        except Exception as e:
            print("snapshot fail:", u, e)
            continue
    if not all_df:
        raise RuntimeError("No TSA snapshots retrieved")
    combined = pd.concat(all_df).reset_index().rename(columns={"index": "date"})
    combined = combined.drop_duplicates(subset="date", keep="first").set_index("date").sort_index()
    combined.to_parquet(cache)
    return combined


def main():
    try:
        tsa = fetch_tsa_history()
    except Exception as e:
        return mark_failed("J11_tsa_throughput", f"TSA scrape failed: {e}")
    if tsa.empty:
        return mark_failed("J11_tsa_throughput", "Empty TSA data")

    s = tsa["throughput"].astype(float).dropna()
    # build 2019 baseline indexed by month-day
    s2019 = s[s.index.year == 2019]
    if len(s2019) < 200:
        return mark_failed("J11_tsa_throughput",
                           f"Insufficient 2019 baseline ({len(s2019)} days)")
    baseline = s2019.copy()
    baseline.index = baseline.index.strftime("%m-%d")
    baseline = baseline.groupby(level=0).mean()

    # series with 2019 same-day baseline for each row
    md = s.index.strftime("%m-%d")
    base_aligned = baseline.reindex(md).values
    s_df = pd.DataFrame({"throughput": s.values, "baseline_2019": base_aligned}, index=s.index)
    s_df["ratio"] = s_df["throughput"] / s_df["baseline_2019"]
    s_df["ratio_7d"] = s_df["ratio"].rolling(7).mean()
    s_df["weekday"] = s_df.index.weekday
    trigger_mask = (s_df["ratio_7d"] < 0.95) & (s_df["weekday"] < 5)  # weekday
    triggers = s_df.index[trigger_mask]

    px = load_prices(["JETS", "SPY"], start="2019-01-01")
    if px.empty or "JETS" not in px.columns:
        return mark_failed("J11_tsa_throughput", "JETS load failed")
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
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    pnl = (daily_pos.shift(1) * rets["JETS"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J11 TSA weak -> short JETS")
    print_metrics(m)
    print(f"\nTSA rows: {len(s)} trigger events (dedup): {n_events}")

    save_result("J11_tsa_throughput", m, extra={
        "status": "ok",
        "rule": "TSA daily throughput 7-day avg < 95% of same-day 2019 baseline on a weekday -> short JETS 30 trading days.",
        "mechanism": "Persistent weakness in passenger volumes vs pre-pandemic baseline indicates demand softness for US airlines.",
        "source": "https://www.tsa.gov/travel/passenger-volumes (current) plus web.archive.org snapshots for 2019-2024 history",
        "n_events": n_events,
        "caveats": "Built from multi-year HTML table cells; baseline of 2019 means strategy structurally inactive once recovery completes.",
    })


if __name__ == "__main__":
    main()
