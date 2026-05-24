"""
Z1 DOJ Deferred Prosecution Agreement (DPA) two-leg signal.

Rule:
- Curated set of 15-20 high-profile DOJ DPAs / NPAs / criminal pleas
  (2010-2024). On the announcement / filing day t, short the issuer
  T+1 for 180 calendar days, then flip and go long T+180 for the next
  120 calendar days. Net pair PnL across all events, equal-weight per
  event.

Mechanism:
- DPAs combine large monetary penalty + ongoing compliance monitor.
  The initial 6 months tend to underperform on penalty drag,
  guidance suspension, and franchise damage. After that, "settlement
  removed" tailwind + analyst rerating + monitor-removal drift drives
  multi-quarter outperformance.

Source:
- DOJ press releases; curated by year / case from public reporting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Curated DPA / criminal-settlement events 2010-2024.
# (ticker, date, description)
EVENTS = [
    ("WFC", "2017-04-20", "Wells Fargo CFPB+OCC autos/sales practices"),
    ("WFC", "2020-02-21", "Wells Fargo $3B DOJ sales practices settlement"),
    ("GS",  "2020-10-22", "Goldman Sachs 1MDB DPA $2.9B"),
    ("BA",  "2021-01-07", "Boeing $2.5B DOJ DPA (737 MAX)"),
    ("CAT", "2017-03-02", "Caterpillar Geneva tax raids / DOJ probe"),
    ("JPM", "2014-01-07", "JPMorgan Madoff $1.7B DPA"),
    ("JPM", "2020-09-29", "JPMorgan spoofing $920M DPA"),
    ("HSBC","2012-12-11", "HSBC $1.9B DPA money laundering"),
    ("CS",  "2014-05-19", "Credit Suisse $2.6B tax-evasion guilty plea"),
    ("BNPQY","2014-06-30","BNP Paribas $8.9B sanctions guilty plea"),
    ("DB",  "2017-01-17", "Deutsche Bank Russian mirror-trades $629M"),
    ("MS",  "2022-09-27", "Morgan Stanley block-trade SEC/DOJ probe disclosed"),
    ("GLEN","2022-05-24", "Glencore $1.1B DOJ FCPA + commodities manipulation guilty plea"),
    ("PFE", "2009-09-02", "Pfizer $2.3B fraud settlement (out of window, kept for breadth)"),
    ("GSK", "2012-07-02", "GSK $3B DOJ settlement"),
    ("VLKAY","2017-01-11","VW $4.3B Dieselgate DOJ plea"),
    ("ERICY","2019-12-06","Ericsson $1.06B FCPA DPA"),
    ("SIEGY","2008-12-15","Siemens FCPA (kept for breadth)"),
    ("BCS", "2015-05-20", "Barclays FX rigging $2.4B guilty plea"),
    ("UBS", "2015-05-20", "UBS FX rigging guilty plea + LIBOR breach"),
    ("RY",  "2018-12-04", "RBC AML probe disclosure"),
    ("TD",  "2024-10-10", "TD Bank $3B AML guilty plea"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())

    try:
        px = load_prices(tickers + ["SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed("Z1_dpa_two_leg", f"price load failed: {e}")

    rets = px.pct_change()
    spy = rets["SPY"]

    # Two legs per event: -1 for ~126 trading days (180 cal), then +1 for ~84 (120 cal).
    SHORT_DAYS = 126
    LONG_DAYS = 84
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]
        d = row["date"]
        if t not in rets.columns:
            n_skipped += 1
            continue
        s = rets[t].dropna()
        if len(s) == 0 or d < s.index[0]:
            n_skipped += 1
            continue
        # find first trading day strictly after d
        nxt = s.index[s.index > d]
        if len(nxt) == 0:
            n_skipped += 1
            continue
        i0 = s.index.get_loc(nxt[0])
        # short leg
        leg1_end = min(i0 + SHORT_DAYS, len(s))
        leg1 = -s.iloc[i0:leg1_end] + spy.reindex(s.index).iloc[i0:leg1_end].fillna(0)
        # long leg (market-neutral via SPY hedge)
        leg2_start = leg1_end
        leg2_end = min(leg2_start + LONG_DAYS, len(s))
        leg2 = s.iloc[leg2_start:leg2_end] - spy.reindex(s.index).iloc[leg2_start:leg2_end].fillna(0)
        combined = pd.concat([leg1, leg2])
        daily_pnls.append(combined)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z1_dpa_two_leg", "no events matched price history")

    # Equal-weight across events: on each calendar day, average the
    # active per-event pnls.
    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z1_dpa_two_leg", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z1 DPA two-leg")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z1_dpa_two_leg", m, extra={
        "status": "ok",
        "rule": ("For each curated DOJ DPA / criminal settlement (2009-2024), "
                 "short issuer (vs SPY) for 126 trading days starting T+1, then "
                 "go long (vs SPY) for the next 84 trading days. Equal-weight "
                 "across overlapping events."),
        "mechanism": ("DPAs penalize first (fines, guidance withdrawal), then "
                      "remove uncertainty for a multi-quarter post-settlement drift."),
        "source": "Curated DOJ press-release dates; public reporting.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
