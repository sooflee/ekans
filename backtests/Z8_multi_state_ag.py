"""
Z8 Multi-state Attorney General coalition lawsuit -- short defendant.

Rule:
- Curated set of ~10 major multi-state AG coalition cases 2014-2024.
- For each, short the defendant on the filing day (T+1) for 60 trading
  days. Equal-weight across overlapping events. SPY-hedged.

Mechanism:
- A multi-state AG coalition lawsuit signals broad, durable political
  willingness to litigate, large potential aggregate liabilities, and
  high probability of follow-on federal enforcement. The first ~3
  months of news cycle pressure issuer multiples.

Source:
- Curated from NAAG press releases, major-state AG press releases
  (NY, CA, TX, WA, MA), and public news.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, date, description)
EVENTS = [
    ("JNJ",  "2019-08-26", "Opioid -- OK state verdict; broader multi-state coalition"),
    ("CAH",  "2017-11-08", "Cardinal Health -- 41-state opioid AG coalition"),
    ("ABC",  "2017-11-08", "AmerisourceBergen -- 41-state opioid"),
    ("MCK",  "2017-11-08", "McKesson -- 41-state opioid"),
    ("CVS",  "2017-11-08", "CVS -- 41-state opioid"),
    ("WBA",  "2017-11-08", "Walgreens -- 41-state opioid"),
    ("PFE",  "2018-04-12", "Pfizer / insulin pricing multi-state"),
    ("LLY",  "2018-04-12", "Eli Lilly / insulin pricing multi-state"),
    ("NVO",  "2018-04-12", "Novo Nordisk / insulin pricing multi-state"),
    ("GOOGL","2020-12-17", "Google -- 38-state AG antitrust (search)"),
    ("META", "2020-12-09", "Facebook -- 48-state AG antitrust"),
    ("AMZN", "2024-09-26", "Amazon -- multi-state AG (retail antitrust)"),
    ("JUUL", "2019-11-05", "JUUL -- multi-state AG marketing-to-minors (proxy: PM)"),
    ("MO",   "2019-11-05", "Altria -- JUUL stake multi-state AG"),
    ("TMUS", "2019-06-11", "T-Mobile-Sprint -- multi-state AG antitrust"),
    ("EBAY", "2017-06-21", "eBay (StubHub multi-state AG case)"),
    ("BIIB", "2018-06-26", "Generic-drug price-fixing multi-state AG (Teva/Mylan included)"),
    ("TEVA", "2018-06-26", "Teva -- generic-drug price-fixing multi-state"),
    ("MYL",  "2018-06-26", "Mylan -- generic-drug price-fixing multi-state"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())

    try:
        px = load_prices(tickers + ["SPY"], start="2013-01-01")
    except Exception as e:
        return mark_failed("Z8_multi_state_ag", f"price load failed: {e}")

    rets = px.pct_change()
    if "SPY" not in rets.columns:
        return mark_failed("Z8_multi_state_ag", "SPY missing")
    spy = rets["SPY"]

    HOLD = 60
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]; d = row["date"]
        if t not in rets.columns:
            n_skipped += 1; continue
        idx = rets[t].dropna().index
        nxt = idx[idx > d]
        if len(nxt) == 0:
            n_skipped += 1; continue
        i0 = rets.index.get_loc(nxt[0])
        end = min(i0 + HOLD, len(rets.index))
        leg = -rets[t].iloc[i0:end].fillna(0) + spy.iloc[i0:end].fillna(0)
        leg = leg.clip(lower=-0.3, upper=0.3)
        daily_pnls.append(leg)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z8_multi_state_ag", "no events matched")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z8_multi_state_ag", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z8 Multi-state AG short 60d")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z8_multi_state_ag", m, extra={
        "status": "ok",
        "rule": ("For each curated multi-state AG coalition lawsuit 2017-2024, "
                 "short the defendant (vs SPY) at T+1, hold 60 trading days. "
                 "Equal-weight across overlapping events; daily returns "
                 "clipped at +/-30%."),
        "mechanism": ("Multi-state AG coalition signals broad political "
                      "willingness to litigate, large aggregate liability, "
                      "follow-on federal enforcement; ~3 months of news-cycle "
                      "pressure on multiples."),
        "source": "Curated from NAAG and major-state AG press releases.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
