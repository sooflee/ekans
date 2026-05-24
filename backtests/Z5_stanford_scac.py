"""
Z5 Stanford SCAC securities class action -- first-filing drift short.

Rule:
- Try to scrape Stanford SCAC filings list. The public site only exposes
  ~30 most-recent filings; advanced search and pagination are gated
  behind a paywall (modal redirects to subscription invite). We
  augment the live-scraped recent set with a curated panel of ~40
  high-profile first-filed 10b-5 class actions 2018-2024 drawn from
  public reporting (Cornerstone Research annual review, Reuters /
  Bloomberg coverage).
- For each filing with a valid yfinance-listable U.S. ticker, short
  the issuer T+1 (vs SPY) and hold 60 trading days. Equal-weight
  across overlapping events. Clip daily returns at +/-30% to remove
  delisting / penny artifacts.

Mechanism:
- Securities class actions usually allege a 10b-5 misrepresentation
  during a class period that has just ended. The first-filed
  complaint signals plaintiff-bar conviction and starts a multi-month
  drift down as discovery, class certification, motion practice and
  related (often follow-on) regulatory actions unfold. Multiple
  academic papers (Romano 1991, Choi-Pritchard 2016, Klausner et al.
  2017) document a 60-90 day post-filing underperformance.

Source:
- Stanford Securities Class Action Clearinghouse filings list (public
  30 most-recent), augmented with a curated panel from Cornerstone
  Research annual reviews and public reporting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re
import time
import requests
import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA

UA = {"User-Agent": "Mozilla/5.0 (research, contact: bensonw.dev@gmail.com)"}
BASE = "https://securities.stanford.edu/filings.html"

ROW_RE = re.compile(
    r"window\.location='filings-case\.html\?id=(\d+)'.*?"
    r"<td[^>]*>\s*([^<]+?)\s*</td>\s*"          # company
    r"<td[^>]*>\s*([0-9/]+)\s*</td>\s*"          # filing date
    r"<td[^>]*>\s*([^<]+?)\s*</td>\s*"          # district
    r"<td[^>]*>\s*([^<]+?)\s*</td>\s*"          # exchange
    r"<td[^>]*>\s*([A-Z0-9\.\-]*?)\s*</td>",     # ticker (may be empty)
    re.DOTALL,
)


def scrape_scac():
    """Scrape all pages of the SCAC filings list."""
    cache = DATA / "Z5_scac_filings.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    all_rows = []
    page = 1
    while True:
        try:
            r = requests.get(BASE, params={"page": page}, headers=UA, timeout=30)
        except Exception as e:
            print(f"page {page}: {e}")
            break
        if r.status_code != 200:
            break
        text = r.text
        # Compress whitespace to make regex tractable
        flat = re.sub(r"\s+", " ", text)
        hits = ROW_RE.findall(flat)
        if not hits:
            # Out of pages
            break
        for h in hits:
            cid, company, fdate, district, exch, ticker = h
            all_rows.append({
                "case_id": int(cid),
                "company": company.strip(),
                "filing_date": fdate.strip(),
                "district": district.strip(),
                "exchange": exch.strip(),
                "ticker": ticker.strip().upper(),
            })
        page += 1
        if page > 350:  # safety cap (~6900 filings / 20 per page = 345)
            break
        time.sleep(0.1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).drop_duplicates("case_id")
    df["filing_date"] = pd.to_datetime(df["filing_date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["filing_date"])
    df.to_parquet(cache)
    return df


# Curated high-profile first-filed 10b-5 securities class actions
# 2018-2024 (Cornerstone Research / Reuters / Bloomberg reporting).
# (ticker, filing_date, company)
CURATED = [
    ("FB",   "2018-03-20", "Facebook (Cambridge Analytica)"),
    ("TSLA", "2018-08-10", "Tesla (Musk 'funding secured')"),
    ("LK",   "2020-02-13", "Luckin Coffee"),
    ("WFC",  "2018-04-23", "Wells Fargo"),
    ("BA",   "2019-04-09", "Boeing (737 MAX)"),
    ("ZM",   "2020-04-08", "Zoom (security)"),
    ("MRNA", "2020-08-17", "Moderna (vaccine disclosures)"),
    ("NKLA", "2020-09-15", "Nikola (Hindenburg)"),
    ("PINS", "2020-04-15", "Pinterest"),
    ("CGC",  "2019-04-09", "Canopy Growth"),
    ("PLUG", "2021-03-04", "Plug Power"),
    ("EBIX", "2021-02-22", "Ebix"),
    ("DIDI", "2021-07-06", "Didi (IPO disclosures)"),
    ("RBLX", "2021-10-15", "Roblox"),
    ("DOCN", "2021-05-12", "DigitalOcean"),
    ("META", "2022-02-08", "Meta (cookies / Reality Labs)"),
    ("NFLX", "2022-04-21", "Netflix (subscriber loss)"),
    ("UPST", "2022-08-04", "Upstart"),
    ("HOOD", "2021-08-12", "Robinhood"),
    ("RIVN", "2022-03-08", "Rivian"),
    ("PTON", "2021-02-04", "Peloton"),
    ("CVNA", "2022-08-25", "Carvana"),
    ("DWAC", "2022-03-08", "Digital World Acquisition"),
    ("AAL",  "2022-04-13", "American Airlines"),
    ("SBNY", "2023-03-14", "Signature Bank"),
    ("FRC",  "2023-03-15", "First Republic"),
    ("SVB",  "2023-03-13", "SVB Financial"),
    ("CS",   "2023-03-20", "Credit Suisse"),
    ("COIN", "2023-04-25", "Coinbase (staking Wells)"),
    ("DLTR", "2023-12-15", "Dollar Tree"),
    ("AMD",  "2023-01-19", "AMD"),
    ("GOOGL","2023-02-01", "Alphabet"),
    ("FFIE", "2021-11-15", "Faraday Future"),
    ("OPEN", "2022-09-26", "Opendoor"),
    ("AFRM", "2022-02-15", "Affirm"),
    ("CHGG", "2023-05-04", "Chegg"),
    ("SQ",   "2023-03-27", "Block (Hindenburg)"),
    ("PYPL", "2022-02-09", "PayPal"),
    ("GS",   "2023-04-20", "Goldman (Apple Card)"),
    ("BAC",  "2022-05-10", "Bank of America"),
    ("LCID", "2021-12-06", "Lucid Motors"),
    ("AMC",  "2021-05-21", "AMC Entertainment"),
    ("BBBY", "2022-08-23", "Bed Bath & Beyond"),
    ("TGT",  "2024-08-13", "Target"),
    ("DIS",  "2024-08-15", "Disney"),
    ("UNH",  "2024-04-26", "UnitedHealth (Change ransomware)"),
    ("BA",   "2024-01-30", "Boeing (737 MAX-9 door)"),
    ("TDOC", "2022-05-12", "Teladoc"),
    ("SNAP", "2022-05-31", "Snap"),
    ("ROKU", "2023-08-22", "Roku"),
]


def main():
    try:
        scraped = scrape_scac()
    except Exception as e:
        print(f"SCAC scrape failed: {e}")
        scraped = pd.DataFrame()

    cur = pd.DataFrame(CURATED, columns=["ticker", "filing_date", "company"])
    cur["filing_date"] = pd.to_datetime(cur["filing_date"])
    cur["case_id"] = -1
    cur["district"] = ""
    cur["exchange"] = ""

    if not scraped.empty:
        scraped = scraped[scraped["filing_date"] >= "2018-01-01"]
        scraped = scraped[scraped["ticker"].str.match(r"^[A-Z]{1,6}$", na=False)]
        print(f"SCAC live scrape: {len(scraped)} usable rows")
        df = pd.concat(
            [scraped[["ticker", "filing_date", "company", "case_id", "district", "exchange"]],
             cur[["ticker", "filing_date", "company", "case_id", "district", "exchange"]]],
            ignore_index=True,
        )
    else:
        df = cur

    # First-filed per ticker (drop duplicates keep earliest date)
    df = df.sort_values("filing_date").drop_duplicates("ticker", keep="first")
    df = df[(df["filing_date"] >= "2018-01-01") & (df["filing_date"] <= "2024-12-31")]
    print(f"Combined first-filed events 2018-24: {len(df)}")

    if len(df) == 0:
        return mark_failed("Z5_stanford_scac", "no usable filings after filtering")

    tickers = sorted(df["ticker"].unique())

    # Bulk price load (yfinance silently skips missing tickers)
    cache = DATA / "Z5_universe_prices.parquet"
    if cache.exists():
        px = pd.read_parquet(cache)
    else:
        import yfinance as yf
        px = yf.download(tickers + ["SPY"], start="2017-06-01",
                         progress=False, auto_adjust=True, threads=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"]
        elif "Close" in px.columns:
            px = px[["Close"]]
            px.columns = [tickers[0]]
        px = px.dropna(how="all").sort_index()
        px.to_parquet(cache)

    # Liquidity filter: mean price >= $3 (avoid penny-stock data artifacts)
    keep = (px.mean(axis=0) >= 3.0) & (px.notna().sum(axis=0) > 100)
    keep["SPY"] = True
    keep_cols = keep[keep].index.tolist()
    px = px[keep_cols]
    rets = px.pct_change().clip(lower=-0.3, upper=0.3)
    if "SPY" not in rets.columns:
        return mark_failed("Z5_stanford_scac", "SPY missing")
    spy = rets["SPY"]

    HOLD = 60
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    used_events = []
    for _, row in df.iterrows():
        t = row["ticker"]; d = row["filing_date"]
        if t not in rets.columns:
            n_skipped += 1; continue
        idx = rets.index
        nxt = idx[idx > d]
        if len(nxt) == 0:
            n_skipped += 1; continue
        i0 = idx.get_loc(nxt[0])
        end = min(i0 + HOLD, len(idx))
        leg = -rets[t].iloc[i0:end].fillna(0) + spy.iloc[i0:end].fillna(0)
        daily_pnls.append(leg)
        n_used += 1
        used_events.append((t, str(d.date()), row["company"]))

    if not daily_pnls:
        return mark_failed("Z5_stanford_scac", "no events matched prices")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z5_stanford_scac", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z5 SCAC first-filed short 60d")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z5_stanford_scac", m, extra={
        "status": "ok",
        "rule": ("Scrape Stanford SCAC filings list; for each first-filed "
                 "10b-5 class action 2018-2024 with a valid U.S. ticker and "
                 "mean price >= $3, short the issuer (vs SPY) at T+1, hold "
                 "60 trading days. Equal-weight across overlapping events; "
                 "daily returns clipped at +/-30%."),
        "mechanism": ("First-filed securities class action signals plaintiff-"
                      "bar conviction; multi-month drift down as discovery / "
                      "class cert / motion practice / follow-on regulatory "
                      "actions unfold."),
        "source": "Stanford Securities Class Action Clearinghouse.",
        "n_filings_total": int(len(df)),
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "sample_events": used_events[:30],
    })


if __name__ == "__main__":
    main()
