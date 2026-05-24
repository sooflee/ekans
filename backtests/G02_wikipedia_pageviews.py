"""
G02 Wikipedia pageview anomalies (also doubles as creative I-11 Wikipedia momentum).

Idea: investor attention spikes show up in Wikipedia article pageviews. Cross-
sectional momentum on month-over-month pageview change. Long top decile, short
bottom decile, equal-weight, monthly rebalance.

Data: Wikimedia Pageviews REST API
  https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/
       all-access/all-agents/<article>/daily/<YYYYMMDD>/<YYYYMMDD>

Basket: ~50 US large-caps with clean Wikipedia article titles.

Honest notes:
- API only covers from 2015-07-01 onward.
- Static survivor-biased basket (same names used in _universe.py).
- We deliberately drop a few names whose Wikipedia titles are ambiguous
  (e.g., "Apple" -> fruit vs company); we use disambiguated titles
  ("Apple_Inc.").
- Position applied at month_t close to month_t+1 returns (no look-ahead).
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import compute_metrics, print_metrics, save_result, mark_failed, DATA
from _universe import load_universe_prices

# Map ticker -> Wikipedia article title (URL-safe form).
TICKER_TO_WIKI = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon_(company)",
    "META": "Meta_Platforms",
    "NVDA": "Nvidia",
    "ORCL": "Oracle_Corporation",
    "CSCO": "Cisco",
    "IBM": "IBM",
    "INTC": "Intel",
    "ADBE": "Adobe_Inc.",
    "QCOM": "Qualcomm",
    "TXN": "Texas_Instruments",
    "AMAT": "Applied_Materials",
    "T": "AT%26T",
    "VZ": "Verizon",
    "CMCSA": "Comcast",
    "DIS": "The_Walt_Disney_Company",
    "NFLX": "Netflix",
    "HD": "The_Home_Depot",
    "LOW": "Lowe%27s",
    "NKE": "Nike,_Inc.",
    "MCD": "McDonald%27s",
    "SBUX": "Starbucks",
    "TGT": "Target_Corporation",
    "COST": "Costco",
    "WMT": "Walmart",
    "PG": "Procter_%26_Gamble",
    "KO": "The_Coca-Cola_Company",
    "PEP": "PepsiCo",
    "JNJ": "Johnson_%26_Johnson",
    "PFE": "Pfizer",
    "MRK": "Merck_%26_Co.",
    "ABT": "Abbott_Laboratories",
    "LLY": "Eli_Lilly_and_Company",
    "UNH": "UnitedHealth_Group",
    "GILD": "Gilead_Sciences",
    "AMGN": "Amgen",
    "JPM": "JPMorgan_Chase",
    "BAC": "Bank_of_America",
    "WFC": "Wells_Fargo",
    "C": "Citigroup",
    "GS": "Goldman_Sachs",
    "MS": "Morgan_Stanley",
    "AXP": "American_Express",
    "BLK": "BlackRock",
    "GE": "General_Electric",
    "BA": "Boeing",
    "CAT": "Caterpillar_Inc.",
    "LMT": "Lockheed_Martin",
    "UPS": "United_Parcel_Service",
    "DE": "John_Deere",
    "XOM": "ExxonMobil",
    "CVX": "Chevron_Corporation",
    "COP": "ConocoPhillips",
    "FCX": "Freeport-McMoRan",
    "NEM": "Newmont",
    "AMT": "American_Tower",
}

API_BASE = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}")
HEADERS = {"User-Agent": "ekans-backtester/0.1 (research; contact: backtest@example.com)"}


def fetch_pageviews(article, start_yyyymmdd, end_yyyymmdd, retries=3):
    """Fetch daily pageviews for one article. Returns a Series indexed by date."""
    url = API_BASE.format(article=article, start=start_yyyymmdd, end=end_yyyymmdd)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                # Article not found; return empty
                return pd.Series(dtype=float, name=article)
            r.raise_for_status()
            data = r.json().get("items", [])
            if not data:
                return pd.Series(dtype=float, name=article)
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["timestamp"].str[:8])
            return df.set_index("date")["views"].astype(float).rename(article)
        except Exception:
            if attempt == retries - 1:
                return pd.Series(dtype=float, name=article)
            time.sleep(2 + attempt * 2)
    return pd.Series(dtype=float, name=article)


def load_all_pageviews(start="20160101", end=None, cache=True):
    """Cache combined pageview DataFrame to data/wiki_pageviews_<n>_<start>_<end>.parquet."""
    import datetime as dt
    if end is None:
        end = dt.date.today().strftime("%Y%m%d")
    fp = DATA / f"wiki_pageviews_{len(TICKER_TO_WIKI)}_{start}_{end}.parquet"
    if cache and fp.exists():
        return pd.read_parquet(fp)
    series_list = []
    for i, (tkr, art) in enumerate(TICKER_TO_WIKI.items()):
        s = fetch_pageviews(art, start, end)
        if len(s) > 0:
            s = s.rename(tkr)
            series_list.append(s)
        # be polite to wikimedia
        time.sleep(0.15)
        if (i + 1) % 10 == 0:
            print(f"  fetched {i+1}/{len(TICKER_TO_WIKI)} articles", flush=True)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1).sort_index()
    if cache:
        df.to_parquet(fp)
    return df


def main():
    try:
        pv = load_all_pageviews(start="20160101")
    except Exception as e:
        return mark_failed("G02_wikipedia_pageviews",
                           f"Wikimedia pageviews fetch failed: {e}")

    if pv.empty or pv.shape[1] < 20:
        return mark_failed("G02_wikipedia_pageviews",
                           f"Too few pageview series fetched: {pv.shape}")

    pv = pv.sort_index()

    # Drop early days with very thin coverage
    coverage = pv.notna().sum(axis=1)
    pv = pv.loc[coverage >= 20]

    # Aggregate to monthly: sum daily views per month
    pv_m = pv.resample("ME").sum(min_count=15)
    # MoM change in pageviews (signal at end of month T, used for month T+1 returns)
    score = pv_m.pct_change()

    # Load price universe for returns
    px = load_universe_prices(start="2015-12-01", include_spy=True)
    spy = px["SPY"]
    universe_cols = [c for c in TICKER_TO_WIKI.keys() if c in px.columns]
    px_u = px[universe_cols].copy()
    rets_d = px_u.pct_change()

    # Monthly compounded returns for performance accounting
    rets_m = (1 + rets_d).resample("ME").prod() - 1
    spy_m = (1 + spy.pct_change()).resample("ME").prod() - 1

    # Align score with universe columns
    score = score[[c for c in score.columns if c in universe_cols]]

    # Decile portfolio construction
    positions = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for d, row in score.iterrows():
        s = row.dropna()
        if len(s) < 20:
            continue
        n_dec = max(3, len(s) // 10)
        top = s.nlargest(n_dec).index
        bot = s.nsmallest(n_dec).index
        positions.loc[d, top] = 1.0 / len(top)
        positions.loc[d, bot] = -1.0 / len(bot)

    # Realized PnL: position at end of month T applied to month T+1 returns
    pos_shift = positions.shift(1)
    aligned_cols = [c for c in pos_shift.columns if c in rets_m.columns]
    pnl_m = (pos_shift[aligned_cols] * rets_m[aligned_cols]).sum(axis=1, min_count=1).dropna()

    if len(pnl_m) < 12:
        return mark_failed("G02_wikipedia_pageviews",
                           f"Too few monthly returns to evaluate ({len(pnl_m)})")

    # Build a synthetic daily PnL by spreading each month's return uniformly
    # over its trading days, so the harness compute_metrics works on a daily series.
    # Alternatively, do a monthly-native metrics calc.
    eq = (1 + pnl_m).cumprod()
    years = len(pnl_m) / 12.0
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl_m.std() * np.sqrt(12)
    sharpe = pnl_m.mean() / pnl_m.std() * np.sqrt(12) if pnl_m.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    max_dd = float(dd.min())
    hit = float((pnl_m > 0).mean())
    t_stat = pnl_m.mean() / (pnl_m.std() / np.sqrt(len(pnl_m))) if pnl_m.std() > 0 else 0

    bench_m = spy_m.reindex(pnl_m.index).dropna()
    bench_cagr = float((1 + bench_m).prod() ** (1 / years) - 1) if len(bench_m) else None
    bench_sharpe = float(bench_m.mean() / bench_m.std() * np.sqrt(12)) if len(bench_m) and bench_m.std() > 0 else None

    metrics = {
        "name": "G02 Wikipedia pageview MoM (long top decile / short bottom decile)",
        "start": str(pnl_m.index[0].date()),
        "end": str(pnl_m.index[-1].date()),
        "n_days": int(len(pnl_m)),
        "n_months": int(len(pnl_m)),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else None,
        "hit_rate": hit,
        "t_stat": float(t_stat),
        "bench_cagr": bench_cagr,
        "bench_sharpe": bench_sharpe,
    }
    if bench_cagr is not None:
        metrics["excess_cagr"] = metrics["cagr"] - bench_cagr
    print_metrics(metrics)
    save_result("G02_wikipedia_pageviews", metrics, extra={
        "status": "ok",
        "rule": "Cross-section: rank universe by MoM Wikipedia pageview change at month-end; "
                "long top decile, short bottom decile, equal-weight, hold 1 month.",
        "universe_size": len(universe_cols),
        "data_source": "Wikimedia REST Pageviews API (2015-07 onward)",
        "frequency": "monthly",
        "caveats": "Survivor-biased static basket; Wikipedia title disambiguation is heuristic; "
                   "no transaction costs; combines with creative I-11 (Wikipedia momentum).",
    })


if __name__ == "__main__":
    main()
