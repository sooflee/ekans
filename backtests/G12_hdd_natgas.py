"""
G12 HDD vs natural gas (CPC daily HDD -> UNG).

Data: NOAA CPC Daily Heating Degree Days (population-weighted, CONUS row),
posted yearly at:
   https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/<YYYY>/Population.Heating.txt

Method:
 1. Pull CONUS daily HDD for 2005-present (or as deep as files cover).
 2. Aggregate to ISO weekly sums.
 3. Each Monday compute current week's HDD-so-far vs prior week's total.
    A simpler reproducible variant: at the close of each Thursday (just
    before the typical EIA storage release Thursday 10:30 ET), compare the
    previous Thu->Wed (= "this past week") HDD to the same window the week
    before. If "this week HDD > prior week HDD * 1.05", go long UNG at the
    next-trading-day open and hold through the FOLLOWING Thursday's close.
 4. Honest about: NG=F is hard to get without slippage; UNG is the public
    instrument; signal is naive (no temperature *forecast* used, only realised).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime as dt
import io
import requests
import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

URL_FMT = ("https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/"
           "daily_data/{year}/Population.Heating.txt")


def fetch_year(year):
    url = URL_FMT.format(year=year)
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    text = r.text
    lines = [ln for ln in text.splitlines() if ln.strip() and "|" in ln]
    if len(lines) < 3:
        return None
    header = None
    for i, ln in enumerate(lines):
        if ln.startswith("Region|") or ln.lower().startswith("region|"):
            header = lines[i]
            data_rows = lines[i+1:]
            break
    if header is None:
        return None
    cols = header.split("|")
    date_cols = cols[1:]
    # Find CONUS row
    conus_row = None
    for r_ in data_rows:
        parts = r_.split("|")
        if parts[0].strip().upper() == "CONUS":
            conus_row = parts[1:]
            break
    if conus_row is None:
        return None
    if len(conus_row) != len(date_cols):
        # truncate to min
        n = min(len(conus_row), len(date_cols))
        conus_row = conus_row[:n]
        date_cols = date_cols[:n]
    s = pd.Series(
        [int(v) if v.strip().lstrip("-").isdigit() else np.nan for v in conus_row],
        index=pd.to_datetime(date_cols, format="%Y%m%d", errors="coerce"),
        name="HDD",
    )
    s = s[s.index.notna()].sort_index()
    return s


def load_all_hdd(start_year=2005, cache=True):
    cache_fp = DATA / f"cpc_hdd_conus_{start_year}.parquet"
    if cache and cache_fp.exists():
        s = pd.read_parquet(cache_fp)["HDD"]
        # Refresh if cache ends more than 7 days ago
        if s.index.max() < pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7):
            cache = False
        else:
            return s
    end_year = dt.date.today().year
    pieces = []
    for y in range(start_year, end_year + 1):
        try:
            s = fetch_year(y)
            if s is not None and not s.empty:
                pieces.append(s)
        except Exception:
            continue
    if not pieces:
        return None
    full = pd.concat(pieces).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    if cache:
        full.to_frame("HDD").to_parquet(cache_fp)
    return full


def main():
    hdd = load_all_hdd(start_year=2005)
    if hdd is None or len(hdd) < 250:
        return mark_failed("G12_hdd_natgas",
                           "Could not assemble CPC HDD CONUS series")

    # Weekly aggregate: use Friday-anchored week-end so we have a full week
    # of observations available by Thursday close.
    weekly_hdd = hdd.resample("W-THU").sum()  # week ending Thursday

    # Signal: this week's HDD vs prior week, evaluated AT thursday week-end.
    wow = weekly_hdd / weekly_hdd.shift(1) - 1
    signal = wow > 0.05  # require >5% WoW jump

    # Trade UNG from Friday open (next session after Thursday EIA release)
    # held until next Thursday close.
    px = load_prices(["UNG"], start="2007-04-19")
    if px.empty or "UNG" not in px.columns:
        return mark_failed("G12_hdd_natgas", "UNG price load failed")
    ung = px["UNG"].dropna()
    ung_rets = ung.pct_change()

    # Build daily position: at each Thursday-end (signal date), if signal True,
    # position +1 for the next ~5 trading days until next Thursday close.
    pos = pd.Series(0.0, index=ung.index)
    for sig_date in signal[signal].index:
        # find next trading day after sig_date
        idx = ung.index.searchsorted(sig_date) + 1
        if idx >= len(ung.index):
            continue
        start = ung.index[idx]
        # hold to next Thursday or until ~7 days later
        end = start + pd.Timedelta(days=7)
        pos.loc[start:end] = 1.0

    pnl = (pos.shift(1) * ung_rets).dropna()
    bench = ung_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="G12 HDD spike -> long UNG")
    print_metrics(m)
    print(f"\nWeeks with HDD spike: {int(signal.sum())} / {int(len(signal.dropna()))}")

    save_result("G12_hdd_natgas", m, extra={
        "status": "ok",
        "rule": ("Weekly CONUS population-weighted HDD; when week WoW > +5%, "
                 "buy UNG at next session open and hold ~1 week."),
        "data_source": "NOAA CPC Daily Heating Degree Days (Population-weighted, CONUS)",
        "n_signal_weeks": int(signal.sum()),
        "n_total_weeks": int(len(signal.dropna())),
        "caveats": ("Uses realised HDD, not 8-14 day forecast (cleaner signal "
                    "would be NDFD 7-day forecast vs prior week). UNG has "
                    "well-known negative roll yield; trade signs may be "
                    "dominated by ETF decay, not weather alpha."),
    })


if __name__ == "__main__":
    main()
