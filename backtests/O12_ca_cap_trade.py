"""
O12 California Cap-and-Trade Auction clearing prices (CARB).
PDF: arb.ca.gov results_summary.pdf, ~52 quarterly joint+CA-only auctions since
Nov 2012.

When current-auction clearing price > 10% above the prior auction's clearing,
long ICLN for 90 trading days.
Mechanism: Auction clearing-price acceleration signals tightening allowance supply
and policy momentum -> bullish for clean-energy equities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


MONTHS = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
          "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}


def fetch_auctions():
    fp = DATA / "carb_auction_clearing.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://ww2.arb.ca.gov/sites/default/files/2020-08/results_summary.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    pdf_path = DATA / "carb_results_summary.pdf"
    with open(pdf_path, "wb") as f:
        f.write(raw)
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() for p in PdfReader(pdf_path).pages)
    # Lines like: "February 2026 Joint Auction #46 54,975,757 54,975,757 $27.94 6,481,750 6,263,000 $27.94"
    # Also: "February 2014 Auction #6 19,538,695 19,538,695 $11.48 ..."
    pat = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\s+(?:Joint\s+)?Auction\s+#(\d+)\s+[\d,]+\s+[\d,]+\s+\$(\d+\.\d{2})"
    )
    rows = []
    for m in pat.finditer(text):
        mo, yr, num, price = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        rows.append((pd.Timestamp(yr, MONTHS[mo], 15), num, price))
    df = pd.DataFrame(rows, columns=["auction_date", "auction_num", "clearing_price"])
    df = df.drop_duplicates(subset=["auction_date","auction_num"]).sort_values("auction_date")
    df = df.set_index("auction_date")
    df.to_parquet(fp)
    return df


def main():
    try:
        auctions = fetch_auctions()
        icln = load_prices(["ICLN"], start="2014-01-01").iloc[:, 0].rename("ICLN")
        spy = load_prices(["SPY"], start="2014-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O12_ca_cap_trade", f"data load failed: {e}")

    print(f"Auctions parsed: {len(auctions)}")
    print(auctions.head(3).to_string())
    print("...")
    print(auctions.tail(3).to_string())

    # Compute QoQ jump
    auctions = auctions.copy()
    auctions["prev_price"] = auctions["clearing_price"].shift(1)
    auctions["pct_chg"] = (auctions["clearing_price"] / auctions["prev_price"] - 1.0)

    triggers = auctions.index[auctions["pct_chg"] > 0.10]
    n_events = len(triggers)

    icln_rets = icln.pct_change()
    pos = pd.Series(0.0, index=icln_rets.index)
    hold = 90
    for d in triggers:
        # ICLN must exist on/after this date
        loc = icln_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 3:
        return mark_failed("O12_ca_cap_trade", f"only {n_events} qualifying events", extra={"n_events": int(n_events)})

    pnl = (pos * icln_rets).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O12 CCA auction +10% -> long ICLN 90d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O12_ca_cap_trade", m, extra={
        "status": "ok",
        "rule": "When CCA auction clearing price > 10% above the prior auction, long ICLN for 90 sessions.",
        "mechanism": "Clearing-price acceleration signals tightening allowance supply + policy momentum -> clean-energy equity tailwind.",
        "universe": "ICLN",
        "source": "California Air Resources Board results_summary.pdf (joint CA-Quebec + CA-only auctions, Nov 2012+).",
        "n_events": int(n_events),
    })


if __name__ == "__main__":
    main()
