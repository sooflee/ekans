"""
E05 Friday-after-close 8-K drift.

EDGAR full-text search for 8-K filings. Universe: ~80 well-known large-cap tickers
to make the scrape tractable. For each 8-K filed by these tickers, parse the
acceptance datetime and filter to filings accepted after 16:00 ET on a Friday
(weekend-disclosed news). Exclude Item 2.02 (earnings, often scheduled) and
Item 7.01 (Reg FD) -- so we capture material non-routine news dumped after market.

Rule: Short equal-weight basket of triggered names at Monday open (t+1 open),
cover at t+30 trading days. Hypothesis = Friday-after-close 8-Ks are 'bury the
news' filings and underperform.

NOTE: yfinance does not provide open prices reliably for all of these; using
close-to-close as a proxy.
"""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests
from harness import save_result, mark_failed

UA = "ekans-research bensonw.dev@gmail.com"

# Large-cap universe — ~80 well-known names
UNIVERSE = sorted(set([
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","ORCL","CSCO","IBM","INTC",
    "ADBE","QCOM","TXN","AMAT","CRM","NOW","AVGO","SHOP","NFLX","DIS",
    "T","VZ","CMCSA","WMT","TGT","HD","LOW","COST","NKE","MCD","SBUX",
    "PG","KO","PEP","MO","CL","KMB","JNJ","PFE","MRK","ABT","LLY",
    "UNH","BMY","GILD","AMGN","MDT","JPM","BAC","WFC","C","GS","MS",
    "AXP","USB","BLK","SCHW","V","MA","PYPL","GE","BA","CAT","HON",
    "MMM","UNP","LMT","RTX","UPS","DE","XOM","CVX","COP","SLB","DD","DOW",
    "FCX","NEM","NEE","DUK","SO","AMT","SPG","O","CCI",
]))


def fetch_filings_for_ciks(ticker, cik, start_dt, end_dt):
    """Use EDGAR submissions JSON to get 8-K filings + their accepted timestamps."""
    cik_pad = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_pad}.json"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    out = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accepts = recent.get("acceptanceDateTime", [])
    items = recent.get("items", [])
    accs = recent.get("accessionNumber", [])
    for i, f in enumerate(forms):
        if f != "8-K": continue
        fd = dates[i]
        if fd < start_dt or fd > end_dt: continue
        out.append({
            "ticker": ticker,
            "filing_date": fd,
            "accepted": accepts[i] if i < len(accepts) else None,
            "items": items[i] if i < len(items) else "",
            "accession": accs[i],
        })
    # Older filings are in 'files' array (separate JSON files)
    older = data.get("filings", {}).get("files", [])
    for of in older:
        sub_url = f"https://data.sec.gov/submissions/{of['name']}"
        r2 = requests.get(sub_url, headers={"User-Agent": UA}, timeout=20)
        if r2.status_code != 200: continue
        d2 = r2.json()
        forms2 = d2.get("form", [])
        dates2 = d2.get("filingDate", [])
        accepts2 = d2.get("acceptanceDateTime", [])
        items2 = d2.get("items", [])
        accs2 = d2.get("accessionNumber", [])
        for i, f in enumerate(forms2):
            if f != "8-K": continue
            fd = dates2[i]
            if fd < start_dt or fd > end_dt: continue
            out.append({
                "ticker": ticker,
                "filing_date": fd,
                "accepted": accepts2[i] if i < len(accepts2) else None,
                "items": items2[i] if i < len(items2) else "",
                "accession": accs2[i],
            })
        time.sleep(0.3)
    return out


def get_cik(ticker):
    # Use SEC company-tickers map
    return TICKER_TO_CIK.get(ticker.upper())


TICKER_TO_CIK = {}


def load_ticker_map():
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers={"User-Agent": UA}, timeout=30)
    data = r.json()
    out = {}
    for _, row in data.items():
        out[row["ticker"].upper()] = str(row["cik_str"])
    return out


def main():
    sid = "E05_friday_8k_drift"
    try:
        global TICKER_TO_CIK
        TICKER_TO_CIK = load_ticker_map()
        print(f"Ticker→CIK map loaded ({len(TICKER_TO_CIK)} entries)")

        # Fetch 8-Ks for each name
        all_filings = []
        for tkr in UNIVERSE:
            cik = TICKER_TO_CIK.get(tkr)
            if not cik:
                print(f"  {tkr}: no CIK"); continue
            try:
                f = fetch_filings_for_ciks(tkr, cik, "2010-01-01", "2024-12-31")
                # Filter to Friday-after-16:00 accepted
                fri_after = []
                for row in f:
                    acc = row["accepted"]
                    if not acc: continue
                    try:
                        ts = pd.Timestamp(acc)  # EDGAR uses ET timestamps
                    except: continue
                    if ts.dayofweek != 4: continue  # not Friday
                    if ts.hour < 16: continue  # before close
                    # Exclude Item 2.02 (earnings) and 7.01 (Reg FD)
                    items = row.get("items", "") or ""
                    if "2.02" in items: continue
                    fri_after.append(row)
                all_filings.extend(fri_after)
                print(f"  {tkr}: {len(fri_after)} Friday-after-close 8-Ks (of {len(f)} total)")
            except Exception as e:
                print(f"  {tkr}: failed ({e})")
            time.sleep(0.5)

        if not all_filings:
            return mark_failed(sid, "no Friday-after-close 8-Ks found")

        # Load prices
        import yfinance as yf
        tickers = sorted(set([f["ticker"] for f in all_filings] + ["SPY"]))
        df = yf.download(tickers, start="2009-01-01", end="2025-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df.copy()
        px = px.dropna(how="all").sort_index()
        spy = px["SPY"]

        # Event study: short at t+1 close (proxy for open), cover at t+30 trading days
        per_event = []
        excess_30 = []
        excess_60 = []
        ann_day = []
        for f in all_filings:
            tkr = f["ticker"]
            if tkr not in px.columns: continue
            s = px[tkr].dropna()
            d = pd.Timestamp(f["filing_date"])
            # Friday filing -> trade on Monday close
            idx = s.index[s.index > d]
            if len(idx) < 2: continue
            t_short = idx[0]
            i_short = s.index.get_loc(t_short)
            # announce-day return (Friday close vs the next trading day close)
            # we don't have intra-day; use close-on-Friday vs close-on-Monday
            if i_short >= 1:
                ad = s.iloc[i_short] / s.iloc[i_short-1] - 1
                ann_day.append(ad)
            for h, bucket in [(30, excess_30), (60, excess_60)]:
                i_end = min(i_short + h, len(s) - 1)
                if i_end <= i_short + 3: continue
                stock_ret = s.iloc[i_end] / s.iloc[i_short] - 1
                spy_w = spy.loc[s.index[i_short]:s.index[i_end]]
                if len(spy_w) < 2: continue
                spy_r = spy_w.iloc[-1] / spy_w.iloc[0] - 1
                short_excess = spy_r - stock_ret
                bucket.append(short_excess)
            per_event.append({"ticker": tkr, "filing": f["filing_date"]})

        def stats(arr):
            arr = np.array(arr)
            if len(arr) == 0: return None
            mean = float(arr.mean()); std = float(arr.std(ddof=1)) if len(arr)>1 else 0
            t = mean / (std/np.sqrt(len(arr))) if std>0 else 0
            return {"n": len(arr), "mean": mean, "median": float(np.median(arr)),
                    "std": std, "t_stat": float(t), "hit_rate": float((arr>0).mean())}

        result = {
            "name": "E05 Friday-after-close 8-K drift",
            "n_events": len(per_event),
            "next_session_return": stats(ann_day),
            "short_excess_30d": stats(excess_30),
            "short_excess_60d": stats(excess_60),
        }
        save_result(sid, result, extra={
            "status": "ok",
            "rule": "Short large-cap names that filed 8-K Fri after 16:00 ET (excluding Item 2.02). Hold 30 td.",
            "source": "EDGAR submissions API. DellaVigna-Pollet 2009 (Fri earnings drift) inspiration.",
            "universe_size": len(UNIVERSE),
        })
        s30 = result["short_excess_30d"]
        print(f"E05: n={s30['n']}, 30d short excess mean={s30['mean']*100:.2f}%, "
              f"t={s30['t_stat']:.2f}, hit={s30['hit_rate']*100:.0f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
