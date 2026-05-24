"""
E07 Schedule 13D activist drift.

Filter EDGAR full-text-search 13D filings to known activist filers (Elliott, Starboard,
Trian, Pershing Square, ValueAct, Engaged Capital). Buy target at t+1 close after
filing date, hold ~18 months (378 trading days). Equal-weight overlapping basket.

Source: Brav-Jiang-Partnoy-Thomas 2008 (JF) found +5-7% CAR around 13D filings with
positive drift over 12-18 months. Replication is mixed in post-2010 era.
"""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests
from harness import save_result, mark_failed

UA = "ekans-research bensonw.dev@gmail.com"

# Known activist filer CIKs (the management entity that signs 13D).
ACTIVISTS = {
    "Elliott Investment Management":      "0001791786",
    "Elliott Associates (legacy)":        "0001632962",  # Elliott Associates LP
    "Paul Singer Elliott Management":     "0000905718",  # Elliott Management Corp
    "Starboard Value":                    "0001517137",
    "Trian Fund Management":              "0001345471",
    "Pershing Square Capital":            "0001336528",
    "ValueAct Capital":                   "0001418814",
    "Engaged Capital":                    "0001551182",
}


def fetch_13d_for_cik(cik, start_dt, end_dt):
    """Return list of dicts: filing_date, issuer_cik, ticker, accession."""
    out = []
    page = 0
    while True:
        url = ("https://efts.sec.gov/LATEST/search-index?q=&forms=SC+13D&"
               f"dateRange=custom&startdt={start_dt}&enddt={end_dt}&"
               f"ciks={cik}&from={page*100}")
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            # Find issuer (non-activist) CIK in source['ciks']
            issuer_cik = None
            issuer_display = None
            for i, c in enumerate(src.get("ciks", [])):
                if c.lstrip("0") == cik.lstrip("0"):
                    continue
                issuer_cik = c
                issuer_display = src.get("display_names", [None]*10)[i]
                break
            ticker = None
            if issuer_display:
                m = re.search(r"\(([A-Z][A-Z0-9\.\-]{0,7})\)\s+\(CIK", issuer_display)
                if m:
                    ticker = m.group(1)
            out.append({
                "filing_date": src.get("file_date"),
                "issuer_cik": issuer_cik,
                "ticker": ticker,
                "form": src.get("form"),
                "accession": src.get("adsh"),
            })
        # paginate
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        page += 1
        if page * 100 >= total:
            break
        if page > 20:  # safety
            break
        time.sleep(0.5)
    return out


def main():
    sid = "E07_activist_13d"
    try:
        # 1. Pull all 13D filings from each activist CIK since 2010
        all_filings = []
        for name, cik in ACTIVISTS.items():
            try:
                rows = fetch_13d_for_cik(cik, "2010-01-01", "2025-12-31")
                # Keep only initial 13D (not amendments)
                rows = [r for r in rows if r["form"] == "SC 13D"]
                print(f"  {name}: {len(rows)} initial 13D filings")
                for r in rows:
                    r["activist"] = name
                all_filings.extend(rows)
            except Exception as e:
                print(f"  {name}: failed ({e})")
            time.sleep(1.0)

        # Filter to those with a ticker we can resolve
        all_filings = [f for f in all_filings if f["ticker"] and f["filing_date"]]
        # Dedupe by (ticker, filing_date)
        seen = set()
        dedup = []
        for f in all_filings:
            k = (f["ticker"], f["filing_date"])
            if k in seen: continue
            seen.add(k)
            dedup.append(f)
        all_filings = dedup
        print(f"Total unique activist-13D filings: {len(all_filings)}")

        if not all_filings:
            return mark_failed(sid, "no activist 13D filings found via EFTS")

        # 2. Load price data
        import yfinance as yf
        tickers = sorted(set([f["ticker"] for f in all_filings] + ["SPY"]))
        # Avoid hammering yfinance: use chunked downloads
        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]
        px_frames = []
        for chunk in chunked(tickers, 50):
            df = yf.download(chunk, start="2009-01-01", end="2025-12-31",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                d = df["Close"]
            else:
                d = df[["Close"]]; d.columns = chunk
            px_frames.append(d)
            time.sleep(0.5)
        px = pd.concat(px_frames, axis=1)
        px = px.loc[:, ~px.columns.duplicated()]
        px = px.dropna(how="all").sort_index()

        if "SPY" not in px.columns:
            return mark_failed(sid, "SPY failed to load")
        spy = px["SPY"]

        # 3. Event study: 18-month hold (378 trading days)
        per_event = []
        excess_18m = []
        excess_12m = []
        excess_6m = []
        for f in all_filings:
            tkr = f["ticker"]
            if tkr not in px.columns: continue
            s = px[tkr].dropna()
            if s.empty: continue
            d = pd.Timestamp(f["filing_date"])
            idx = s.index[s.index >= d]
            if len(idx) < 2: continue
            t0 = idx[0]
            i0 = s.index.get_loc(t0)
            i_buy = i0 + 1
            if i_buy >= len(s) - 5: continue
            for h, bucket in [(126, excess_6m), (252, excess_12m), (378, excess_18m)]:
                i_end = min(i_buy + h, len(s) - 1)
                if i_end <= i_buy + 5: continue
                stock_ret = s.iloc[i_end] / s.iloc[i_buy] - 1
                spy_w = spy.loc[s.index[i_buy]:s.index[i_end]]
                if len(spy_w) < 2: continue
                spy_r = spy_w.iloc[-1] / spy_w.iloc[0] - 1
                bucket.append(stock_ret - spy_r)
            per_event.append({"ticker": tkr, "filing": f["filing_date"], "activist": f["activist"]})

        def stats(arr):
            arr = np.array(arr)
            if len(arr) == 0: return None
            mean = float(arr.mean()); std = float(arr.std(ddof=1)) if len(arr)>1 else 0
            t = mean / (std/np.sqrt(len(arr))) if std>0 else 0
            return {"n": len(arr), "mean": mean, "median": float(np.median(arr)),
                    "std": std, "t_stat": float(t), "hit_rate": float((arr>0).mean())}

        result = {
            "name": "E07 Activist 13D drift",
            "n_filings_resolved": len(per_event),
            "excess_6m":  stats(excess_6m),
            "excess_12m": stats(excess_12m),
            "excess_18m": stats(excess_18m),
        }
        save_result(sid, result, extra={
            "status": "ok",
            "rule": "Buy 13D target at t+1 close, hold 18m (378 td), equal-weight overlapping basket.",
            "source": "EDGAR EFTS full-text search filtered to activist CIKs. Brav-Jiang-Partnoy-Thomas 2008.",
            "activists": list(ACTIVISTS.keys()),
        })
        s18 = result["excess_18m"]
        print(f"E07: n={s18['n']}, 18m excess mean={s18['mean']*100:.2f}%, "
              f"t={s18['t_stat']:.2f}, hit={s18['hit_rate']*100:.0f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
