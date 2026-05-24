"""
W4 Treasury auction tail → short long-duration risk.

Mechanism: When 10y or 30y Treasury auctions "tail" (stop-out yield prints
materially above the when-issued / median), it signals dealers had to take
down supply at a haircut. Long-duration-sensitive assets (utilities, REITs,
long-duration tech / ARKK) tend to underperform in the session and into
the next day as the rates move feeds through.

Source: TreasuryDirect.gov auction-results API
        https://www.treasurydirect.gov/TA_WS/securities/auctioned

Proxy for "tail":
  The TD API does NOT publish the when-issued yield directly. We use the
  spread between highYield and averageMedianYield (also called the "tail to
  median") as a free proxy. Auctions where (high - median) > 1.5 bp are
  flagged as TAIL events.

Rule (per spec): on auction day t with tail > 1.5 bp, short an equal-weight
basket of XLU + XLRE + ARKK from t+1 open to t+1 close (using daily close-to-
close return as a proxy since we only have daily bars).

Long-only baseline: hold the negative of the basket return on event-day+1 only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import requests
import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned"


def fetch_auctions(security_type, days_back=3650):
    """Fetch auction results for a given security type from TreasuryDirect."""
    params = {"format": "json", "type": security_type, "days": days_back}
    r = requests.get(TD_URL, params=params, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.DataFrame(r.json())


def main():
    cache_fp = DATA / "treasurydirect_auctions_W4.parquet"
    if cache_fp.exists():
        try:
            auctions = pd.read_parquet(cache_fp)
        except Exception:
            auctions = None
    else:
        auctions = None

    if auctions is None:
        try:
            notes = fetch_auctions("Note", 3650)
            time.sleep(0.5)
            bonds = fetch_auctions("Bond", 3650)
        except Exception as e:
            return mark_failed("W4_auction_tail_long_dur",
                               f"TreasuryDirect fetch failed: {e}")
        auctions = pd.concat([notes, bonds], ignore_index=True)
        try:
            auctions.to_parquet(cache_fp)
        except Exception:
            pass

    if auctions.empty:
        return mark_failed("W4_auction_tail_long_dur",
                           "TreasuryDirect returned empty")

    # Keep 10y and 30y, original-issue or reopenings
    auctions["auctionDate"] = pd.to_datetime(auctions["auctionDate"],
                                              errors="coerce")
    auctions["highYield"] = pd.to_numeric(auctions["highYield"], errors="coerce")
    auctions["averageMedianYield"] = pd.to_numeric(
        auctions["averageMedianYield"], errors="coerce")
    auctions["bidToCoverRatio"] = pd.to_numeric(
        auctions["bidToCoverRatio"], errors="coerce")

    keep_terms = ["10-Year", "29-Year 10-Month", "30-Year",
                   "29-Year 11-Month", "9-Year 10-Month", "9-Year 11-Month"]
    # Use originalSecurityTerm in {10-Year, 30-Year}
    auctions["origTerm"] = auctions["originalSecurityTerm"].fillna("")
    sel = auctions[auctions["origTerm"].isin(["10-Year", "30-Year"])].copy()
    sel = sel.dropna(subset=["auctionDate", "highYield", "averageMedianYield"])
    sel["tail_bp"] = (sel["highYield"] - sel["averageMedianYield"]) * 100.0  # in bp
    # Filter to auctions on or after 2018-01-01
    sel = sel[sel["auctionDate"] >= "2018-01-01"]
    # Drop pathological negatives (rare data errors)
    sel = sel[sel["tail_bp"] > -10]

    # Event: tail > 1.5 bp
    tail_events = sel[sel["tail_bp"] > 1.5].copy()
    tail_events = tail_events.sort_values("auctionDate")
    if len(tail_events) < 5:
        return mark_failed("W4_auction_tail_long_dur",
                           f"Only {len(tail_events)} tail events found")

    # Load price data for basket
    basket = ["XLU", "XLRE", "ARKK"]
    try:
        px = load_prices(basket + ["SPY"], start="2017-06-01")
    except Exception as e:
        return mark_failed("W4_auction_tail_long_dur",
                           f"Price load failed: {e}")

    # Many of the early XLRE/ARKK data start later; drop missing
    px = px.dropna(how="any")
    rets = px.pct_change()
    spy_ret = rets["SPY"]
    basket_ret = rets[basket].mean(axis=1)  # equal-weight basket return

    # For each event date, compute next-day return of (short basket) = -basket_ret
    event_pnls = []
    for ev_date in tail_events["auctionDate"]:
        # Find next trading day
        future = rets.index[rets.index > ev_date]
        if len(future) == 0:
            continue
        nxt = future[0]
        if nxt not in basket_ret.index:
            continue
        # Pair: short basket, long SPY beta-hedge — but spec says just short the basket.
        # We'll also report SPY-hedged version.
        event_pnls.append({
            "event_date": ev_date,
            "next_date": nxt,
            "tail_bp": float(tail_events.loc[tail_events["auctionDate"] == ev_date,
                                              "tail_bp"].iloc[0]),
            "basket_ret": float(basket_ret.loc[nxt]),
            "spy_ret": float(spy_ret.loc[nxt]),
            "short_basket_ret": float(-basket_ret.loc[nxt]),
            "short_basket_vs_spy": float(-basket_ret.loc[nxt] + spy_ret.loc[nxt]),
        })

    ev_df = pd.DataFrame(event_pnls).set_index("next_date").sort_index()
    if len(ev_df) < 5:
        return mark_failed("W4_auction_tail_long_dur",
                           f"Too few events after price alignment: {len(ev_df)}")

    # Build daily pnl series at event dates; on non-event days pnl=0
    pnl_short = ev_df["short_basket_ret"]
    # Reindex onto full daily calendar so compute_metrics handles correctly
    full_idx = rets.index[rets.index >= ev_df.index.min()]
    pnl_full = pd.Series(0.0, index=full_idx)
    for d, v in pnl_short.items():
        if d in pnl_full.index:
            pnl_full.loc[d] = v

    bench = spy_ret.reindex(pnl_full.index)
    m = compute_metrics(pnl_full, benchmark=bench, name="W4 auction tail short XLU+XLRE+ARKK")
    m["n_events"] = int(len(ev_df))
    m["mean_event_return"] = float(ev_df["short_basket_ret"].mean())
    m["median_event_return"] = float(ev_df["short_basket_ret"].median())
    m["event_hit_rate"] = float((ev_df["short_basket_ret"] > 0).mean())
    m["pair_mean_vs_spy"] = float(ev_df["short_basket_vs_spy"].mean())
    print_metrics(m)
    print(f"Events: {len(ev_df)}, "
          f"mean short-basket return: {m['mean_event_return']*100:.3f}%, "
          f"hit rate: {m['event_hit_rate']*100:.1f}%")

    save_result("W4_auction_tail_long_dur", m, extra={
        "status": "ok",
        "rule": ("On a 10y or 30y Treasury auction day where the tail "
                 "(highYield - averageMedianYield) exceeds 1.5 bp, short an "
                 "equal-weight XLU+XLRE+ARKK basket on the next trading day "
                 "(daily close-to-close)."),
        "mechanism": ("A tail signals weaker dealer demand at the auction; "
                       "rates step up which compresses long-duration equity "
                       "valuations (utilities, REITs, long-tech)."),
        "source": ("TreasuryDirect.gov auction results API. Tail proxy = "
                    "highYield − averageMedianYield (the published 'tail to "
                    "median'). The true 'tail' (vs when-issued) is similar "
                    "in spirit but is not in the public TD JSON."),
        "universe": "XLU, XLRE, ARKK (equal-weight short basket); SPY benchmark.",
        "n_events": int(len(ev_df)),
        "tail_threshold_bp": 1.5,
    })


if __name__ == "__main__":
    main()
