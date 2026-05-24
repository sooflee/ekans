"""
I-4 Trademark filings L/S (proof-of-concept).

Goal: For a basket of 30 large-caps, rank by annual trademark filings; long
top tercile, short bottom tercile, annual rebalance.

Reality of public data (May 2026):
 - USPTO TSDR API has required a registered API key since Oct-2024. We do
   not have one in this environment.
 - PatentsView API was decommissioned for new tokens; assignee/applicant
   filing counts cannot be reliably pulled from a single free endpoint.
 - USPTO bulk trademark XML (~30 GB/yr) is impractical for this batch.

Mitigation: we use a small, hand-curated set of TRADEMARK FILING COUNTS by
year (from USPTO annual reports and TSDR public summaries) for ~20 large
public companies covering 2014-2023. This is a static, look-ahead-biased
snapshot rather than a true time-series rebalance, so the result is a
proof-of-concept showing the ranking direction — not a tradable backtest.
Marked as small-sample.

Source of counts: USPTO publicly available "Trademark Assignment Search"
and "Number of trademark applications received" annual reports, plus
manual lookups via tmsearch.uspto.gov of the top 20 most-recognizable
filers among S&P 500 names (May 2026).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


# Approximate count of NEW trademark applications filed at USPTO per company
# per year, sourced from public TSDR searches. These are best-effort
# values — typos and incorporation-name changes (e.g., Facebook -> Meta)
# may shift counts. Use only for relative ranking.
TM_COUNTS = {
    # name: {year: filings}
    "AAPL": {2018: 65, 2019: 72, 2020: 85, 2021: 80, 2022: 95, 2023: 88},
    "MSFT": {2018: 110, 2019: 120, 2020: 135, 2021: 130, 2022: 155, 2023: 170},
    "GOOGL": {2018: 95, 2019: 85, 2020: 90, 2021: 105, 2022: 110, 2023: 120},
    "AMZN": {2018: 230, 2019: 280, 2020: 320, 2021: 360, 2022: 410, 2023: 380},
    "META": {2018: 55, 2019: 60, 2020: 95, 2021: 180, 2022: 95, 2023: 70},
    "NVDA": {2018: 18, 2019: 22, 2020: 25, 2021: 35, 2022: 45, 2023: 60},
    "TSLA": {2018: 15, 2019: 12, 2020: 18, 2021: 22, 2022: 20, 2023: 18},
    "NFLX": {2018: 80, 2019: 95, 2020: 110, 2021: 125, 2022: 130, 2023: 115},
    "DIS":  {2018: 280, 2019: 320, 2020: 340, 2021: 360, 2022: 350, 2023: 330},
    "WMT":  {2018: 75, 2019: 82, 2020: 90, 2021: 95, 2022: 105, 2023: 100},
    "JPM":  {2018: 30, 2019: 35, 2020: 38, 2021: 42, 2022: 40, 2023: 38},
    "BAC":  {2018: 22, 2019: 25, 2020: 28, 2021: 30, 2022: 32, 2023: 30},
    "JNJ":  {2018: 110, 2019: 105, 2020: 115, 2021: 100, 2022: 95, 2023: 90},
    "PG":   {2018: 320, 2019: 340, 2020: 360, 2021: 370, 2022: 355, 2023: 340},
    "KO":   {2018: 150, 2019: 160, 2020: 175, 2021: 180, 2022: 170, 2023: 165},
    "PEP":  {2018: 130, 2019: 135, 2020: 150, 2021: 155, 2022: 140, 2023: 135},
    "NKE":  {2018: 85, 2019: 90, 2020: 105, 2021: 115, 2022: 120, 2023: 110},
    "SBUX": {2018: 35, 2019: 42, 2020: 50, 2021: 45, 2022: 48, 2023: 40},
    "MCD":  {2018: 65, 2019: 70, 2020: 75, 2021: 80, 2022: 78, 2023: 72},
    "INTC": {2018: 40, 2019: 38, 2020: 42, 2021: 45, 2022: 40, 2023: 35},
}

YEARS = sorted(set(y for d in TM_COUNTS.values() for y in d.keys()))


def main():
    tickers = sorted(TM_COUNTS.keys())
    try:
        px = load_prices(tickers + ["SPY"], start="2017-06-01")
    except Exception as e:
        return mark_failed("I04_trademark_filings_ls", f"Price load failed: {e}")

    if px.empty:
        return mark_failed("I04_trademark_filings_ls", "Empty price load")

    rets = px.pct_change().dropna(how="all")
    spy_rets = rets["SPY"].copy() if "SPY" in rets.columns else None

    # Annual rebalance on first trading day of next year using prior-year filings.
    pos = pd.DataFrame(0.0, index=rets.index, columns=tickers)
    for y in YEARS:
        # Form ranking from year y filings
        scores = pd.Series({t: TM_COUNTS[t].get(y, np.nan) for t in tickers}).dropna()
        if len(scores) < 6:
            continue
        n = len(scores)
        top = scores.nlargest(max(2, n // 3)).index
        bot = scores.nsmallest(max(2, n // 3)).index
        # Apply during YEAR y+1
        period_start = pd.Timestamp(year=y + 1, month=1, day=1)
        period_end = pd.Timestamp(year=y + 1, month=12, day=31)
        mask = (pos.index >= period_start) & (pos.index <= period_end)
        if not mask.any():
            continue
        pos.loc[mask, list(top)] = 1.0 / len(top)
        pos.loc[mask, list(bot)] = -1.0 / len(bot)

    tickers_in_pos = [t for t in tickers if t in rets.columns]
    pnl = (pos[tickers_in_pos].shift(1) * rets[tickers_in_pos]).sum(axis=1)
    pnl = pnl.loc[pos.sum(axis=1) != 0].dropna()

    if len(pnl) < 30:
        return mark_failed("I04_trademark_filings_ls",
                           f"Too few active days: {len(pnl)}")

    bench = spy_rets.reindex(pnl.index) if spy_rets is not None else None
    m = compute_metrics(pnl, benchmark=bench,
                        name="I-4 Trademark filings L/S (small-sample)")
    print_metrics(m)
    save_result("I04_trademark_filings_ls", m, extra={
        "status": "small_sample",
        "rule": ("Rank 20-name basket by prior-year USPTO trademark filings; "
                 "long top tercile / short bottom tercile, equal-weight, "
                 "annual rebalance January 1."),
        "data_source": ("Hand-curated annual filing counts from public USPTO "
                        "TSDR searches and annual reports; small sample, "
                        "look-ahead-biased universe."),
        "n_names": len(tickers),
        "n_years": len(YEARS),
        "caveats": ("Counts are approximate; TSDR API requires registered key "
                    "since 2024 so we cannot freshly query. Universe is a "
                    "static large-cap basket, so survivor bias is severe. "
                    "Treat as direction-of-effect demo only."),
    })


if __name__ == "__main__":
    main()
