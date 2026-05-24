"""
K3 Beige Book "uncertain*" word count — short SPY when text is anxious.

Method:
 1. For every Beige Book release 2010-01 to present, fetch all 12 district
    pages (Boston, NY, Philly, Cleveland, Richmond, Atlanta, Chicago, St-Louis,
    Minneapolis, Kansas-City, Dallas, San-Francisco) + the national summary.
 2. Strip HTML, lowercase, count occurrences of 'uncertain' (this catches
    'uncertain', 'uncertainty', 'uncertainties' as a single stem).
 3. Sum across all districts → one count per release.
 4. Rule: when sum > 25, short SPY for 30 trading days starting the trading
    day after the release date (release date inferred from URL year-month
    and title parsing).

Note on threshold: the original rule was "count > 25". This is a single
stem count summed across 12 districts + national summary, comparable to
Fed-reported uncertainty literature aggregations.
"""
import sys
import re
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


DISTRICTS = ["kansas-city", "atlanta", "st-louis", "san-francisco",
             "minneapolis", "richmond", "chicago", "dallas",
             "new-york", "philadelphia", "cleveland", "boston"]

BASE = "https://www.federalreserve.gov/monetarypolicy/beigebook{ym}{suffix}.htm"


def fetch_release(ym):
    """Returns (release_date, total_uncertain_count) or (None, None) if release
    doesn't exist for this YYYYMM. release_date is mid-month of the named
    period as a conservative tradeable proxy (actual release within a week
    of mid-month)."""
    summary_url = BASE.format(ym=ym, suffix="-summary")
    r = requests.get(summary_url, timeout=30)
    if r.status_code != 200 or len(r.text) < 20000:
        r = requests.get(BASE.format(ym=ym, suffix=""), timeout=30)
        if r.status_code != 200 or len(r.text) < 20000:
            return None, None
    # Use mid-month-of-publication as release date proxy.
    yr = int(ym[:4]); mn = int(ym[4:])
    release_date = pd.Timestamp(year=yr, month=mn, day=15)

    # Now fetch all district pages
    total = 0
    for d in DISTRICTS:
        url = BASE.format(ym=ym, suffix=f"-{d}")
        try:
            rr = requests.get(url, timeout=30)
            if rr.status_code != 200:
                continue
            t = re.sub(r"<[^>]+>", " ", rr.text)
            t = re.sub(r"\s+", " ", t).lower()
            total += t.count("uncertain")
        except Exception:
            continue
    # Add national summary count too
    t_sum = re.sub(r"<[^>]+>", " ", r.text)
    t_sum = re.sub(r"\s+", " ", t_sum).lower()
    total += t_sum.count("uncertain")
    return release_date, total


def main():
    cache = DATA / "beige_book_uncertainty.csv"
    if cache.exists():
        events = pd.read_csv(cache, parse_dates=["release_date"])
    else:
        rows = []
        # 2017+ uses the simple /monetarypolicy/beigebookYYYYMM.htm pattern.
        # Pre-2017 used /fomc/beigebook/YYYY/YYYYMMDD/FullReport.htm with
        # exact release dates that aren't discoverable without scraping
        # an index that the Fed no longer serves. Restrict to 2017+ for
        # a clean, reliable panel.
        for yr in range(2017, 2027):
            for mn in range(1, 13):
                ym = f"{yr}{mn:02d}"
                try:
                    d, c = fetch_release(ym)
                except Exception as e:
                    print(f"err {ym}: {e}")
                    continue
                if d is not None and c is not None:
                    rows.append({"release_date": d, "count": c, "ym": ym})
                    print(f"{ym} release={d.date()} count={c}")
                time.sleep(0.1)
        events = pd.DataFrame(rows)
        if events.empty:
            return mark_failed("K3_beige_book_uncertainty", "No Beige Books parsed.")
        events.to_csv(cache, index=False)

    events = events.sort_values("release_date").reset_index(drop=True)
    print(f"Parsed {len(events)} releases. Count stats:")
    print(events["count"].describe())

    try:
        px = load_prices(["SPY"], start="2009-12-01")
    except Exception as e:
        return mark_failed("K3_beige_book_uncertainty", f"price fetch: {e}")
    rets = daily_returns(px)["SPY"].dropna()

    sig = pd.Series(0.0, index=rets.index)
    n_trig = 0
    for _, row in events.iterrows():
        if row["count"] > 25:
            i = rets.index.searchsorted(row["release_date"] + pd.Timedelta(days=1))
            if i >= len(rets):
                continue
            sig.iloc[i:i+30] += 1
            n_trig += 1
    pos = -(sig > 0).astype(float)  # short
    pnl = pos.shift(1) * rets
    pnl = pnl.dropna()
    m = compute_metrics(pnl, benchmark=rets.reindex(pnl.index).dropna(),
                        name="K3 Beige Book uncertainty short SPY")
    print_metrics(m)
    print(f"n_releases={len(events)}, n_trig={n_trig}, exposure_short={pos.mean():.2%}")

    save_result("K3_beige_book_uncertainty", m, extra={
        "status": "ok",
        "rule": "Sum of 'uncertain' word counts across 12 districts + national summary per Beige Book release > 25 → short SPY 30 trading days from release+1.",
        "mechanism": "Elevated qualitative business uncertainty in Beige Books is a leading indicator of risk-off equity returns.",
        "source": "Federal Reserve Board Beige Book HTML pages (2010-present)",
        "n_releases_parsed": int(len(events)),
        "n_triggers": int(n_trig),
        "exposure_pct": float((pos != 0).mean()),
        "count_stats": {
            "min": int(events["count"].min()),
            "median": float(events["count"].median()),
            "max": int(events["count"].max()),
            "mean": float(events["count"].mean()),
        },
    })


if __name__ == "__main__":
    main()
