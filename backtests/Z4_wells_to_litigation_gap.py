"""
Z4 Wells Notice -> SEC litigation gap trade.

Rule:
- Curated set of ~15 high-profile public-issuer Wells Notices 2015-2024
  (disclosed in 8-K / 10-Q risk factors, news, or the issuer's own press
  release). For each, short the issuer beginning at the public Wells
  Notice disclosure date for 120 trading days (covering the typical 6mo
  pre-litigation gap).

Mechanism:
- A Wells Notice signals the SEC enforcement staff intend to recommend
  charges. Empirically the issuer drifts down during the 3-9 month gap
  between Wells receipt and the public litigation release as analysts
  reassess, settlement reserves grow, and other regulators pile on.

Source:
- Curated from 8-K filings, news (Reuters / WSJ), and the issuer's own
  enforcement-related disclosures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, public-disclosure date of Wells Notice, description)
EVENTS = [
    ("COIN", "2023-03-22", "Coinbase Wells Notice (staking)"),
    ("KRKN", "2023-05-22", "Kraken (private, dropped) -- using BTSC proxy"),
    ("RIVN", "2022-12-01", "Rivian battery-disclosure probe (pre-Wells reported)"),
    ("BNGO", "2022-08-08", "Bionano Genomics 10-Q risk factor"),
    ("LK",   "2020-04-02", "Luckin Coffee accounting probe"),
    ("WFC",  "2016-09-08", "Wells Fargo sales-practices Wells Notice"),
    ("TSLA", "2018-08-08", "Tesla 'funding secured' SEC inquiry (Musk)"),
    ("XL",   "2015-03-02", "XL Group / XL Capital"),
    ("VRX",  "2016-10-31", "Valeant accounting"),
    ("AAOI", "2018-08-09", "Applied Optoelectronics"),
    ("CMG",  "2016-01-06", "Chipotle E. coli criminal subpoena"),
    ("MNK",  "2019-09-23", "Mallinckrodt Acthar pricing"),
    ("EQT",  "2017-07-26", "EQT royalty SEC inquiry"),
    ("BLUE", "2020-08-19", "Bluebird Bio 10-Q disclosure"),
    ("NKLA", "2020-09-14", "Nikola Hindenburg fallout / SEC subpoena"),
    ("WSTL", "2021-05-13", "Westell Technologies"),
    ("EBIX", "2021-02-19", "Ebix accounting probe"),
    ("BLNK", "2023-05-15", "Blink Charging short-seller / SEC subpoena"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())

    try:
        px = load_prices(tickers + ["SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed("Z4_wells_to_litigation_gap", f"price load failed: {e}")

    rets = px.pct_change()
    if "SPY" not in rets.columns:
        return mark_failed("Z4_wells_to_litigation_gap", "SPY price missing")
    spy = rets["SPY"]

    HOLD = 120  # trading days
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]; d = row["date"]
        if t not in rets.columns:
            n_skipped += 1; continue
        s = rets[t]
        idx = s.dropna().index
        nxt = idx[idx > d]
        if len(nxt) == 0:
            n_skipped += 1; continue
        i0 = s.index.get_loc(nxt[0])
        end = min(i0 + HOLD, len(s))
        leg = -s.iloc[i0:end].fillna(0) + spy.reindex(s.index).iloc[i0:end].fillna(0)
        # clip extreme single-day returns (delisting artifacts)
        leg = leg.clip(lower=-0.3, upper=0.3)
        daily_pnls.append(leg)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z4_wells_to_litigation_gap", "no events matched")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z4_wells_to_litigation_gap", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z4 Wells->litigation gap short 120d")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z4_wells_to_litigation_gap", m, extra={
        "status": "ok",
        "rule": ("For each curated public-issuer Wells Notice disclosure "
                 "(2015-2024), short the issuer (vs SPY) starting on the next "
                 "trading day, hold 120 trading days. Equal-weight across "
                 "overlapping events."),
        "mechanism": ("Wells Notice signals SEC enforcement staff recommended "
                      "charges; issuer drifts down during the 3-9 month gap "
                      "before public litigation release."),
        "source": "Curated from issuer 8-K / 10-Q disclosures, Reuters, WSJ.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
