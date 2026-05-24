"""
L6 SEC NT 10-K / NT 10-Q late-filing notifications -> short the filer.

Rule:
- For each NT 10-K or NT 10-Q filing 2015-present (EDGAR full-text search),
  map the filer CIK to a public ticker. If the ticker has yfinance data,
  short the stock starting at filing date + 1 trading day, cover at
  + 30 calendar days. Equal-weight across overlapping positions.

Mechanism:
- A Form NT signals the registrant could not file the periodic report on
  time. Empirically this is associated with accounting issues, going-
  concern doubts, or auditor friction; the stock tends to underperform
  in the following weeks.

Source:
- EDGAR full-text search index https://efts.sec.gov/LATEST/search-index
- CIK<->ticker map https://www.sec.gov/files/company_tickers.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import requests
import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

UA = {"User-Agent": "ekans research benson@example.com"}


def pull_nt_filings():
    cache = DATA / "sec_nt10_filings.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    base = "https://efts.sec.gov/LATEST/search-index"
    rows = []
    # Iterate per quarter to keep result sets under the 10k EDGAR cap
    quarters = pd.date_range("2015-01-01", "2025-12-31", freq="QS")
    for qstart in quarters:
        qend = qstart + pd.offsets.QuarterEnd(0)
        for form in ("NT 10-K", "NT 10-Q"):
            params = {
                "q": "",
                "dateRange": "custom",
                "startdt": qstart.strftime("%Y-%m-%d"),
                "enddt": qend.strftime("%Y-%m-%d"),
                "forms": form,
            }
            offset = 0
            while True:
                params["from"] = offset
                r = requests.get(base, params=params, headers=UA, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for h in hits:
                    src = h.get("_source", {})
                    ciks = src.get("ciks") or []
                    if not ciks:
                        continue
                    rows.append({
                        "cik": int(ciks[0]),
                        "form": src.get("form"),
                        "file_date": src.get("file_date"),
                        "adsh": src.get("adsh"),
                        "display_name": (src.get("display_names") or [None])[0],
                    })
                if len(hits) < 100:
                    break
                offset += 100
                time.sleep(0.1)
            time.sleep(0.1)
    df = pd.DataFrame(rows).drop_duplicates(["adsh"])
    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")
    df = df.dropna(subset=["file_date"])
    df.to_parquet(cache)
    return df


def cik_ticker_map():
    cache = DATA / "sec_cik_tickers.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(list(j.values()))
    df["cik"] = df["cik_str"].astype(int)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[["cik", "ticker"]].drop_duplicates("cik")
    df.to_parquet(cache)
    return df


def main():
    try:
        filings = pull_nt_filings()
    except Exception as e:
        return mark_failed("L6_sec_nt_10k", f"EDGAR fetch: {e}")
    if filings.empty:
        return mark_failed("L6_sec_nt_10k", "No NT 10-K/10-Q filings parsed")

    try:
        ticks = cik_ticker_map()
    except Exception as e:
        return mark_failed("L6_sec_nt_10k", f"CIK->ticker fetch: {e}")

    merged = filings.merge(ticks, on="cik", how="inner")
    merged = merged.sort_values("file_date").reset_index(drop=True)
    print(f"NT filings parsed: {len(filings)} ; with public ticker: {len(merged)}")

    universe = sorted(merged["ticker"].unique())
    # Sanity cap to avoid hammering yfinance with delisted/non-existent tickers
    print(f"Tickers: {len(universe)}")

    # Bulk price load with a generous start date, batch to keep parquet manageable
    cache = DATA / "L6_universe_prices.parquet"
    if cache.exists():
        px = pd.read_parquet(cache)
    else:
        # try to fetch; missing tickers will be silently dropped by yfinance
        import yfinance as yf
        px = yf.download(universe, start="2014-06-01", progress=False,
                         auto_adjust=True, threads=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"]
        elif "Close" in px.columns:
            px = px[["Close"]]
            px.columns = [universe[0]]
        px = px.dropna(how="all").sort_index()
        px.to_parquet(cache)

    # Filter to tickers with reasonable price history: mean price >= $5 and >250 trading days
    # (excludes penny stocks with corrupted yfinance data and most very thinly traded names)
    keep = (px.mean(axis=0) >= 5.0) & (px.notna().sum(axis=0) > 250)
    px = px.loc[:, keep]
    rets = px.pct_change()
    # Clip extreme single-day returns at ±30% (true delisting drops -> -100% are also artifacts here)
    rets = rets.clip(lower=-0.3, upper=0.3)
    # SPY as benchmark
    spy = load_prices("SPY", start="2014-06-01")["SPY"].pct_change()

    # Build event-driven PnL: each NT filing -> short ticker from t+1 over next 21 trading days (≈30 calendar)
    # Use a SET of active (ticker,event) pairs each day. Position per active ticker = -1 (cap, not stacking).
    # Daily PnL = mean across currently-active positions (equal-weight by ticker, not by event-count).
    HOLD = 21
    n_events = 0
    skipped = 0
    # build a sparse per-event window list, then flatten per day
    events = []  # list of (start_idx, ticker)
    for _, row in merged.iterrows():
        t = row["ticker"]
        d = row["file_date"]
        if t not in rets.columns:
            skipped += 1
            continue
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            skipped += 1
            continue
        i = rets.index.get_loc(nxt[0])
        events.append((i, t))
        n_events += 1

    # For each day, tickers active = {t : any event with i<=day<i+HOLD}.
    active_per_day = [set() for _ in range(len(rets.index))]
    for (i, t) in events:
        for j in range(i, min(i + HOLD, len(rets.index))):
            active_per_day[j].add(t)

    pnl_vals = []
    pnl_idx = []
    # PnL on day d uses positions from day d-1 (shift(1))
    rets_np = rets.values
    cols = {c: k for k, c in enumerate(rets.columns)}
    for d in range(1, len(rets.index)):
        actives = active_per_day[d - 1]
        if not actives:
            continue
        rs = []
        for t in actives:
            if t in cols:
                v = rets_np[d, cols[t]]
                if not np.isnan(v):
                    rs.append(v)
        if rs:
            pnl_vals.append(-np.mean(rs))  # short
            pnl_idx.append(rets.index[d])
    pnl = pd.Series(pnl_vals, index=pnl_idx)

    if len(pnl) < 30:
        return mark_failed("L6_sec_nt_10k",
                           f"insufficient overlap with yfinance prices (active days={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="L6 SEC NT 10-K/Q short 21d")
    print_metrics(m)
    print(f"\nEvents matched to prices: {n_events}, skipped (no price): {skipped}")
    print(f"Active trading days: {len(pnl)}")

    save_result("L6_sec_nt_10k", m, extra={
        "status": "ok",
        "rule": ("For each NT 10-K / NT 10-Q filing on EDGAR (2015-present) "
                 "with a public ticker, short the issuer on the next trading "
                 "day for 21 trading days. Equal-weight by ticker across "
                 "currently-active events. Filter universe to mean price >= $5 "
                 "and clip daily returns at +/-30% to remove penny-stock data artifacts."),
        "mechanism": "Form NT signals delayed periodic reporting -> often coincides with accounting issues / going-concern / audit friction; stocks underperform in following weeks.",
        "source": "EDGAR full-text search-index API + company_tickers.json",
        "n_events": int(n_events),
        "n_filings_with_ticker": int(len(merged)),
        "active_days": int(len(pnl)),
    })


if __name__ == "__main__":
    main()
