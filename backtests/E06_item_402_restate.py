"""
E06 Item 4.02 non-reliance restatement short.

EDGAR full-text search for 8-K filings with "Item 4.02" (notice that previously
issued financial statements should no longer be relied upon). For each, extract
issuer ticker, short at t+1 open (approx close), cover at t+252 (12 months).
Mean CAR vs SPY.

Source: Hennes-Leone-Miller 2008 (TAR), Plumlee-Yohn 2010 — Item 4.02 events
are highly negative-news (avg -10% to -20% announcement-day drop), with continued
post-event drift documented at 6-12 month horizons.
"""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests
from harness import save_result, mark_failed

UA = "ekans-research bensonw.dev@gmail.com"


def fetch_402_filings(start_dt="2010-01-01", end_dt="2024-12-31", max_pages=80):
    out = []
    page = 0
    while page < max_pages:
        url = ("https://efts.sec.gov/LATEST/search-index?q=%22Item+4.02%22&forms=8-K&"
               f"dateRange=custom&startdt={start_dt}&enddt={end_dt}&from={page*100}")
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            print(f"  page {page}: {r.status_code} stop")
            break
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            items = src.get("items") or []
            if "4.02" not in items and not any("4.02" in str(i) for i in items):
                continue
            displays = src.get("display_names", [])
            if not displays:
                continue
            ticker = None
            issuer_cik = None
            for i, d in enumerate(displays):
                m = re.search(r"\(([A-Z][A-Z0-9\.\-]{0,7})\)\s+\(CIK", d)
                if m:
                    ticker = m.group(1)
                    issuer_cik = src["ciks"][i] if i < len(src.get("ciks",[])) else None
                    break
            out.append({
                "filing_date": src.get("file_date"),
                "ticker": ticker,
                "issuer_cik": issuer_cik,
                "form": src.get("form"),
                "items": items,
            })
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        page += 1
        if page * 100 >= total:
            break
        time.sleep(0.3)  # be polite
    return out


def main():
    sid = "E06_item_402_restate"
    try:
        print("Fetching Item 4.02 filings from EDGAR EFTS ...")
        filings = fetch_402_filings()
        print(f"Total filings: {len(filings)}")
        # Keep only those with a ticker (so we can get price data)
        with_t = [f for f in filings if f["ticker"]]
        print(f"With resolvable ticker: {len(with_t)}")
        if not with_t:
            return mark_failed(sid, "no Item 4.02 filings had resolvable tickers")

        # Dedupe (ticker, filing_date)
        seen = set(); clean = []
        for f in with_t:
            k = (f["ticker"], f["filing_date"])
            if k in seen: continue
            seen.add(k); clean.append(f)
        with_t = clean
        print(f"After dedupe: {len(with_t)}")

        # Load prices
        import yfinance as yf
        tickers = sorted(set([f["ticker"] for f in with_t] + ["SPY"]))
        # chunked download
        px_frames = []
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i+100]
            df = yf.download(chunk, start="2009-01-01", end="2025-12-31",
                             progress=False, auto_adjust=True, threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                d = df["Close"]
            else:
                d = df[["Close"]].rename(columns={"Close": chunk[0]})
            px_frames.append(d)
            time.sleep(0.5)
        px = pd.concat(px_frames, axis=1)
        px = px.loc[:, ~px.columns.duplicated()]
        px = px.dropna(how="all").sort_index()
        if "SPY" not in px.columns:
            return mark_failed(sid, "SPY price load failed")
        spy = px["SPY"]

        # Event study: short at t+1 close, cover at t+252
        per_event = []
        excess_252 = []   # excess of short (= -stock_ret - (-spy_ret) = spy_ret - stock_ret)
        excess_126 = []
        excess_21 = []
        announce_day = []
        for f in with_t:
            tkr = f["ticker"]
            if tkr not in px.columns: continue
            s = px[tkr].dropna()
            if len(s) < 50: continue
            d = pd.Timestamp(f["filing_date"])
            idx = s.index[s.index >= d]
            if len(idx) < 2: continue
            t0 = idx[0]
            i0 = s.index.get_loc(t0)
            i_short = i0 + 1
            if i_short >= len(s) - 5: continue
            # announce-day return (0 to +1)
            if i0 >= 1:
                ad = s.iloc[i0] / s.iloc[i0 - 1] - 1
                announce_day.append(ad)
            for h, bucket in [(21, excess_21), (126, excess_126), (252, excess_252)]:
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
            "name": "E06 Item 4.02 restatement short",
            "n_events": len(per_event),
            "announce_day_return": stats(announce_day),
            "short_excess_21d":  stats(excess_21),
            "short_excess_126d": stats(excess_126),
            "short_excess_252d": stats(excess_252),
        }
        save_result(sid, result, extra={
            "status": "ok",
            "rule": "Short ticker at t+1 close after 8-K Item 4.02; cover at t+252; excess vs short SPY.",
            "source": "EDGAR EFTS Item 4.02 8-K search. Hennes-Leone-Miller 2008, Plumlee-Yohn 2010.",
        })
        s252 = result["short_excess_252d"]
        print(f"E06: n={s252['n']}, 252d short excess mean={s252['mean']*100:.2f}%, "
              f"t={s252['t_stat']:.2f}, hit={s252['hit_rate']*100:.0f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
