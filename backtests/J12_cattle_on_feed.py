"""
J12 USDA Cattle on Feed placements low decile -> long LE (use COW ETF / LIVESTOCK proxy).

USDA NASS QuickStats requires an API key. Instead we scrape the public Cattle
on Feed text releases (cofdMMYY.txt) via:
  - Live: usda.library.cornell.edu releases page (last few months)
  - Wayback: web.archive.org snapshots of downloads.usda.library.cornell.edu for 2018-2024
  - Older: usda.mannlib.cornell.edu legacy host (via wayback) for 2014-2017

Rule:
- Extract monthly "Placements during <Month>" figure (current year) from each release.
- Compute MoM % change. When MoM % change is in the bottom decile of historical sample,
  go long the Invesco DB Agriculture Fund's cattle proxy COW (if missing, use COW.SI? else
  fall back to FCT futures proxy via 'COW' ticker family) -- we'll use 'COW' first, else 'LE=F'.
- Hold 30 trading days.
"""
import sys
import re
import json
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


def harvest_cofd_urls():
    """Build a comprehensive list of cofd txt URLs from current + wayback CDX."""
    urls = set()

    # current
    try:
        r = requests.get("https://usda.library.cornell.edu/concern/publications/m326m174z?locale=en",
                         timeout=30, headers=UA)
        for href in re.findall(r'href="(/sites/default/release-files/[^"]+\.txt)"', r.text):
            urls.add("https://usda.library.cornell.edu" + href)
    except Exception as e:
        print("current page err:", e)

    # downloads.usda.library.cornell.edu via wayback CDX
    try:
        r = requests.get(
            "https://web.archive.org/cdx/search/cdx?url=downloads.usda.library.cornell.edu/usda-esmis/files/m326m174z/*&from=20100101&to=20251231&filter=mimetype:text/plain&output=json&limit=2000",
            timeout=120,
        )
        if r.status_code == 200:
            data = json.loads(r.text)
            for row in data[1:]:
                orig = row[2]
                if "cofd" in orig and orig.endswith(".txt"):
                    urls.add(orig)
    except Exception as e:
        print("downloads CDX err:", e)

    # legacy mannlib via wayback
    try:
        r = requests.get(
            "https://web.archive.org/cdx/search/cdx?url=usda.mannlib.cornell.edu/usda/current/CattOnFe/*&from=20050101&to=20180101&filter=mimetype:text/plain&output=json&limit=2000",
            timeout=120,
        )
        if r.status_code == 200:
            data = json.loads(r.text)
            for row in data[1:]:
                orig = row[2]
                if orig.endswith(".txt"):
                    urls.add(orig)
    except Exception as e:
        print("mannlib CDX err:", e)

    return sorted(urls)


def fetch_with_fallback(url):
    """Try direct fetch; if 404 or DNS, fall back to wayback if-available."""
    try:
        r = requests.get(url, timeout=30, headers=UA)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception:
        pass
    # wayback API
    try:
        api = f"https://archive.org/wayback/available?url={url}"
        j = requests.get(api, timeout=30).json()
        snap = j.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available"):
            wb = snap["url"]
            r = requests.get(wb, timeout=60, headers=UA)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
    except Exception:
        pass
    return None


MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]


def parse_release(text):
    """Extract (placement_month_date, placements_1000_head) from a COF release.

    'Placements in feedlots during <Month> totaled X.XX million head'
    Use whatever year is in the release; placement_month is that <Month> in
    the previous calendar (or stated year context).
    """
    # narrative version
    m = re.search(r"[Pp]lacements\s+in\s+feedlots\s+during\s+([A-Za-z]+)\s+totaled\s+([\d\.,]+)\s+million\s+head", text)
    if not m:
        # alt phrasing
        m = re.search(r"[Pp]lacements\s+during\s+([A-Za-z]+)\s+totaled\s+([\d\.,]+)\s+million\s+head", text)
    if not m:
        return None, None
    month_name = m.group(1).capitalize()
    val_str = m.group(2).replace(",", "")
    try:
        val_mn = float(val_str)
    except ValueError:
        return None, None
    if month_name not in MONTH_NAMES:
        return None, None
    month_num = MONTH_NAMES.index(month_name) + 1

    # find year: look near phrase "Released ... 20YY" or first 4-digit year on the page
    yr_match = re.search(r"Released[^\d]+\d{1,2},?\s+(\d{4})", text)
    if yr_match:
        release_year = int(yr_match.group(1))
    else:
        yr_match = re.search(r"(20[12]\d)", text)
        if not yr_match:
            return None, None
        release_year = int(yr_match.group(1))

    # placement month is for the month BEFORE the release month;
    # if release_month is January, placement_month is December of release_year-1.
    # we infer the placement year as release_year if month_num < release_month, else release_year-1.
    # But simpler: extract release month from filename if available -- not available here.
    # Use heuristic: placement_year = release_year if month_num < (release_year_month from "Released" string)
    rel_full = re.search(r"Released\s+([A-Za-z]+)\s+\d{1,2},?\s+(\d{4})", text)
    if rel_full:
        rel_month_name = rel_full.group(1).capitalize()
        if rel_month_name in MONTH_NAMES:
            rel_month_num = MONTH_NAMES.index(rel_month_name) + 1
            if month_num < rel_month_num:
                placement_year = release_year
            else:
                placement_year = release_year - 1
        else:
            placement_year = release_year - 1
    else:
        placement_year = release_year - 1

    try:
        placement_date = pd.Timestamp(year=placement_year, month=month_num, day=1)
    except ValueError:
        return None, None

    return placement_date, val_mn * 1000.0  # in 1000 head


def build_placements_series():
    cache = DATA / "usda_cattle_placements.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    urls = harvest_cofd_urls()
    print(f"Harvested {len(urls)} candidate URLs")
    records = {}
    for i, u in enumerate(urls):
        txt = fetch_with_fallback(u)
        if not txt:
            continue
        d, v = parse_release(txt)
        if d is None or v is None:
            continue
        # if multiple releases reference same placement month, prefer the first (most timely)
        if d not in records:
            records[d] = v
        if (i + 1) % 25 == 0:
            print(f"  parsed {i+1}/{len(urls)}; have {len(records)} unique months")
    if not records:
        raise RuntimeError("No placement values parsed")
    df = pd.Series(records, name="placements_k").sort_index().to_frame()
    df.to_parquet(cache)
    return df


def main():
    try:
        df = build_placements_series()
    except Exception as e:
        return mark_failed("J12_cattle_on_feed", f"COF scrape failed: {e}")

    placements = df["placements_k"].astype(float).dropna()
    if len(placements) < 24:
        return mark_failed("J12_cattle_on_feed",
                           f"Insufficient placement history ({len(placements)} months)")

    mom = placements.pct_change()
    decile = mom.dropna().quantile(0.10)
    triggers = mom[mom < decile].dropna().index

    # Trade COW (iPath cattle ETF) first; fall back to live cattle futures front-month proxy
    px = None
    used_ticker = None
    for tk in ["COW", "LE=F"]:
        try:
            p = load_prices([tk], start="2014-01-01")
            if not p.empty and tk in p.columns and p[tk].dropna().shape[0] > 200:
                px = p
                used_ticker = tk
                break
        except Exception:
            continue
    if px is None:
        # fall back to SPY for a return scaffold and mark failed
        return mark_failed("J12_cattle_on_feed",
                           "No cattle ETF/futures price available (tried COW, LE=F)")

    rets = px[used_ticker].pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in triggers:
        # placement month d is the prior calendar month; release is mid-next month;
        # we know about it only AFTER release date. Approx release ~22nd of month after d.
        signal_avail = d + pd.offsets.MonthEnd(1) + pd.Timedelta(days=22)
        nxt = rets.index[rets.index >= signal_avail]
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
    # benchmark: buy-and-hold of same ticker
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name=f"J12 Low-decile placements -> long {used_ticker}")
    print_metrics(m)
    print(f"\nPlacements n={len(placements)} ; triggers={n_events} ; ticker={used_ticker}")

    save_result("J12_cattle_on_feed", m, extra={
        "status": "ok",
        "rule": (f"USDA Cattle-on-Feed monthly placements in bottom decile of historical MoM "
                 f"change -> long {used_ticker} for 30 trading days post release."),
        "mechanism": "Low placements imply tighter future fed-cattle supply (5-8 months out), supporting live cattle prices.",
        "source": "USDA Cattle on Feed text releases via usda.library.cornell.edu + wayback CDX for legacy archive (https://web.archive.org/cdx/search/cdx?url=downloads.usda.library.cornell.edu/usda-esmis/files/m326m174z/*).",
        "n_events": n_events,
        "n_placement_months": int(len(placements)),
        "ticker_used": used_ticker,
        "caveats": "Smaller sample than full history (NASS API skipped); placement-year inferred from 'Released' date heuristic.",
    })


if __name__ == "__main__":
    main()
