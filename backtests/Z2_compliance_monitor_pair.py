"""
Z2 Compliance monitor pair-trade.

Rule:
- Same DPA events as Z1. For each, short the issuer / long the sector
  ETF for the (hard-coded) length of the compliance monitor, typically
  3 years (756 trading days). Equal-weight across events.

Mechanism:
- During the active monitorship the issuer carries (a) capital-allocation
  constraints, (b) elevated legal-reserve drag, and (c) capped pricing
  power vs sector peers. We test whether the issuer underperforms its
  GICS sub-industry ETF over the full 3-year monitor term.

Source:
- DOJ press releases (monitor length is typically disclosed in the
  Statement of Facts / DPA agreement). Hard-coded 3y where length not
  explicit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, sector ETF, date, monitor years, description)
EVENTS = [
    ("WFC",  "XLF", "2017-04-20", 3, "Wells Fargo CFPB"),
    ("WFC",  "XLF", "2020-02-21", 3, "Wells Fargo DOJ $3B"),
    ("GS",   "XLF", "2020-10-22", 3, "Goldman 1MDB"),
    ("BA",   "ITA", "2021-01-07", 3, "Boeing DPA 737 MAX"),
    ("CAT",  "XLI", "2017-03-02", 2, "Caterpillar tax probe"),
    ("JPM",  "XLF", "2014-01-07", 2, "JPM Madoff"),
    ("JPM",  "XLF", "2020-09-29", 3, "JPM spoofing"),
    ("HSBC", "XLF", "2012-12-11", 5, "HSBC AML (5y monitor)"),
    ("CS",   "XLF", "2014-05-19", 3, "Credit Suisse tax"),
    ("BNPQY","XLF", "2014-06-30", 3, "BNP sanctions"),
    ("DB",   "XLF", "2017-01-17", 3, "Deutsche Bank mirror trades"),
    ("MS",   "XLF", "2022-09-27", 3, "Morgan Stanley block trade"),
    ("GLEN", "XLE", "2022-05-24", 3, "Glencore FCPA"),
    ("GSK",  "XLV", "2012-07-02", 5, "GSK CIA"),
    ("VLKAY","XLI", "2017-01-11", 3, "VW Dieselgate"),
    ("ERICY","XLK", "2019-12-06", 3, "Ericsson FCPA"),
    ("BCS",  "XLF", "2015-05-20", 3, "Barclays FX"),
    ("UBS",  "XLF", "2015-05-20", 3, "UBS FX"),
    ("TD",   "XLF", "2024-10-10", 5, "TD Bank AML"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "etf", "date", "years", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(set(df["ticker"]).union(df["etf"]).union(["SPY"]))

    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed("Z2_compliance_monitor_pair", f"price load failed: {e}")

    rets = px.pct_change()
    spy = rets["SPY"]
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]; e = row["etf"]; d = row["date"]
        y = int(row["years"])
        if t not in rets.columns or e not in rets.columns:
            n_skipped += 1
            continue
        idx = rets.index
        nxt = idx[idx > d]
        if len(nxt) == 0:
            n_skipped += 1
            continue
        i0 = idx.get_loc(nxt[0])
        hold = min(252 * y, len(idx) - i0)
        if hold < 60:
            n_skipped += 1
            continue
        # short issuer, long sector ETF
        leg = -rets[t].iloc[i0:i0 + hold].fillna(0) + rets[e].iloc[i0:i0 + hold].fillna(0)
        daily_pnls.append(leg)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z2_compliance_monitor_pair", "no events matched")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z2_compliance_monitor_pair", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z2 Compliance monitor pair")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z2_compliance_monitor_pair", m, extra={
        "status": "ok",
        "rule": ("For each curated DPA, short the issuer / long the GICS "
                 "sub-industry ETF for the duration of the compliance monitor "
                 "(typically 3 years, hard-coded). Equal-weight across events."),
        "mechanism": ("During an active DOJ monitorship the issuer carries "
                      "elevated legal reserves, capital constraints, and "
                      "compliance overhead vs sector peers."),
        "source": "Curated from DOJ press releases / DPA agreements.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
