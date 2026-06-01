"""PL749_spinoff_stub_parent_long — Spin-Off When-Issued -> Long Stub Parent"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL749_spinoff_stub_parent_long"

    # Hand-coded spin-off completion dates (when-issued to regular-way transition)
    # entry = date of distribution / first regular-way trading day
    events = [
        # (parent_ticker, spinoff_date, spinoff_name)
        ("GE",  "2023-01-03", "GE HealthCare spin-off"),
        ("GE",  "2024-04-02", "GE Vernova spin-off"),
        ("MMM", "2024-04-01", "3M Solventum spin-off"),
        ("T",   "2022-04-11", "AT&T WarnerMedia spin-off"),
        ("IBM", "2021-11-03", "IBM Kyndryl spin-off"),
        ("DD",  "2021-02-01", "DuPont Nutrition & Biosciences spin-off"),
        ("JNJ", "2023-05-04", "J&J Kenvue IPO/distribution"),
        ("PFE", "2023-01-01", "N/A - no recent spin"),  # placeholder, will skip if missing events
        ("WRK", "2020-03-01", "N/A"),  # remove invalid
        ("MRK", "2021-06-02", "Merck Organon spin-off"),
    ]

    # Keep only confirmed large-cap spin-offs with known dates
    confirmed_events = [
        ("GE",  "2023-01-03", "GE HealthCare spin-off"),
        ("GE",  "2024-04-02", "GE Vernova spin-off"),
        ("MMM", "2024-04-01", "3M Solventum spin-off"),
        ("T",   "2022-04-11", "AT&T WarnerMedia spin-off"),
        ("IBM", "2021-11-03", "IBM Kyndryl spin-off"),
        ("MRK", "2021-06-02", "Merck Organon spin-off"),
        ("JNJ", "2023-05-04", "J&J Kenvue"),
        ("WBA", "2021-10-01", "Walgreens AmerisourceBergen - not a spinoff, skip"),
        ("DIS", "2021-03-01", "Disney ESPN placeholder"),  # use only real ones
        ("PFE", "2020-11-16", "Pfizer Upjohn/Viatris spin-off"),
    ]

    # Cleaned list
    confirmed_events = [
        ("GE",  "2023-01-03"),
        ("GE",  "2024-04-02"),
        ("MMM", "2024-04-01"),
        ("T",   "2022-04-11"),
        ("IBM", "2021-11-03"),
        ("MRK", "2021-06-02"),
        ("JNJ", "2023-05-04"),
        ("PFE", "2020-11-16"),
        ("DD",  "2021-02-01"),
        ("BDX", "2022-04-01"),   # BD spin-off of Embecta
    ]

    tickers_needed = list(set([t for t, _ in confirmed_events] + ["SPY"]))

    try:
        px = load_prices(tickers_needed, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    r = daily_returns(px)
    spy_r = r["SPY"]

    hold = 15  # trading days
    pnl = pd.Series(0.0, index=spy_r.index)
    event_details = []
    skipped = 0

    for ticker, spinoff_date_str in confirmed_events:
        spinoff_dt = pd.Timestamp(spinoff_date_str)

        if ticker not in r.columns:
            skipped += 1
            continue

        ticker_r = r[ticker]
        mask = ticker_r.index >= spinoff_dt
        if mask.sum() < hold:
            skipped += 1
            continue

        ei = ticker_r.index[mask][0]
        p = ticker_r.index.get_loc(ei)
        ep = min(p + hold, len(ticker_r))

        seg = ticker_r.iloc[p:ep]
        spy_seg = spy_r.reindex(seg.index).fillna(0)
        excess = seg.values - spy_seg.values

        pnl.iloc[p:ep] += excess[:ep - p]

        parent_ret = float((1 + seg).prod() - 1)
        spy_ret_event = float((1 + spy_seg.iloc[:ep - p]).prod() - 1)
        event_details.append({
            "ticker": ticker,
            "spinoff_date": spinoff_date_str,
            "entry_date": str(ei.date()),
            "parent_return": round(parent_ret, 4),
            "spy_return": round(spy_ret_event, 4),
            "excess_return": round(parent_ret - spy_ret_event, 4),
        })

    print(f"Events processed: {len(event_details)}, skipped: {skipped}")

    if len(event_details) == 0:
        return mark_failed(sid, "no valid events")

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")

    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Spin-Off When-Issued -> Long Stub Parent")
    save_result(sid, m, extra={
        "rule": "On spin-off distribution date, long stub parent for 15 trading days",
        "mechanism": "Post-spin selling pressure on stub parent as index funds rebalance and arb flows clear; mean reversion opportunity",
        "source": "Hand-coded spin-off completion dates; yfinance",
        "n_events": len(event_details),
        "n_skipped": skipped,
        "events": event_details,
    })
    print(f"Done: Sharpe={m.get('sharpe', '?'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
