"""
L3 Treasury 10Y/30Y auction direct-bidder share -> short TLT.

Rule:
- For each 10Y Note or 30Y Bond auction (including reopenings), compute the
  direct-bidder share = direct_bidder_accepted / total_accepted.
- When the direct share at auction is < 5%, short TLT for 5 trading sessions
  starting the session after the auction date.

Mechanism:
- A weak direct-bidder participation at long-end auctions signals soft
  domestic demand and tends to coincide with a near-term sell-off in long
  Treasuries (rates higher / TLT lower) as primary dealers absorb supply.

Source:
- Treasury fiscaldata auctions_query
  https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
       "accounting/od/auctions_query")


def pull_auctions():
    cache = DATA / "treasury_auctions_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    fields = ("auction_date,issue_date,security_type,security_term,reopening,cusip,"
              "direct_bidder_accepted,indirect_bidder_accepted,primary_dealer_accepted,"
              "total_accepted,total_tendered,bid_to_cover_ratio,high_yield,offering_amt")
    page = 1
    page_size = 1000
    while True:
        url = (f"{API}?fields={fields}"
               f"&page%5Bnumber%5D={page}&page%5Bsize%5D={page_size}"
               f"&sort=auction_date")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        page += 1
    df = pd.DataFrame(rows)
    df.to_parquet(cache)
    return df


def main():
    try:
        df = pull_auctions()
    except Exception as e:
        return mark_failed("L3_auction_direct_bidder", f"fiscaldata fetch: {e}")
    if df.empty:
        return mark_failed("L3_auction_direct_bidder", "Empty auctions dataset.")

    # Filter to 10Y Notes (incl reopenings) and 30Y Bonds
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    for col in ("direct_bidder_accepted", "total_accepted"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (
        ((df["security_type"] == "Note") & (df["security_term"] == "10-Year"))
        | ((df["security_type"] == "Bond") & (df["security_term"] == "30-Year"))
    )
    aux = df[mask & df["auction_date"].notna() & (df["total_accepted"] > 0)].copy()
    aux["direct_share"] = aux["direct_bidder_accepted"] / aux["total_accepted"]
    aux = aux.sort_values("auction_date")

    px = load_prices(["TLT", "SPY"], start="2003-01-01")
    rets = px.pct_change()

    pos = pd.Series(0.0, index=rets.index)
    triggers = aux[aux["direct_share"] < 0.05]
    n_events = 0
    for d in triggers["auction_date"]:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start_idx = rets.index.get_loc(nxt[0])
        end_idx = min(start_idx + 5, len(rets.index))
        for j in range(start_idx, end_idx):
            pos.iloc[j] = -1.0
        n_events += 1

    pnl = (pos.shift(1) * rets["TLT"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="L3 Auction direct < 5% -> short TLT 5d")
    print_metrics(m)
    print(f"\nAuctions 10/30Y: {len(aux)} ; trigger events: {n_events}")
    print(f"Median direct share: {aux['direct_share'].median():.3f}")

    save_result("L3_auction_direct_bidder", m, extra={
        "status": "ok",
        "rule": ("10Y or 30Y auction direct-bidder share (direct_bidder_accepted / "
                 "total_accepted) < 5% -> short TLT for 5 trading sessions."),
        "mechanism": "Weak direct demand at long-end auctions implies tail risk in long bond pricing.",
        "source": "Treasury fiscaldata auctions_query API",
        "n_auctions": int(len(aux)),
        "n_triggers": int(n_events),
        "median_direct_share": float(aux["direct_share"].median()),
    })


if __name__ == "__main__":
    main()
