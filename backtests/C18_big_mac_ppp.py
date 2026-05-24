"""
C18 Big Mac PPP
Pull The Economist's Big Mac index raw data. Test mean-reversion:
twice yearly, sort G10 currencies (USD reference) by Big Mac PPP deviation.
Long top-quintile undervalued vs short top-quintile overvalued (use FX ETFs as proxies).
1 year hold per ranking.

Free FX ETF coverage: FXE (EUR), FXY (JPY), FXB (GBP), FXF (CHF), FXC (CAD), FXA (AUD).
That's 6 currencies vs USD — too few for quintiles. We adapt: long the top-3 most undervalued,
short the top-3 most overvalued (i.e., 50/50 split) at each rebalance, equal weight.
"""
import sys
import io
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    DATA, load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ETF_MAP = {
    "EUZ": "FXE",
    "JPN": "FXY",
    "GBR": "FXB",
    "CHE": "FXF",
    "CAN": "FXC",
    "AUS": "FXA",
}


def fetch_big_mac():
    cache = DATA / "big_mac_full.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    url = ("https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/"
           "output-data/big-mac-full-index.csv")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    df.to_csv(cache, index=False)
    return df


def main():
    try:
        bm = fetch_big_mac()
    except Exception as e:
        return mark_failed("C18_big_mac_ppp", f"could not fetch Big Mac CSV: {e}")

    bm = bm[bm["iso_a3"].isin(ETF_MAP.keys())].copy()
    if bm.empty:
        return mark_failed("C18_big_mac_ppp", "no overlap between Big Mac universe and free FX ETFs")

    # USD_raw column: PPP-implied deviation vs USD. Negative => undervalued; positive => overvalued.
    bm = bm[["date", "iso_a3", "USD_raw"]].dropna()
    bm = bm.sort_values("date")

    etf_tickers = list(set(ETF_MAP.values()))
    try:
        px = load_prices(etf_tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed("C18_big_mac_ppp", f"FX ETF load failed: {e}")
    px = px.dropna()

    # For each Big Mac release date, build target weights and hold until next release (~6 months).
    weights_at_release = []
    for d, grp in bm.groupby("date"):
        scored = grp.set_index("iso_a3")["USD_raw"]
        scored = scored.reindex(ETF_MAP.keys()).dropna()
        if len(scored) < 4:
            continue
        ranks = scored.rank()
        n = len(scored)
        k = max(2, n // 3)  # top/bottom third
        undervalued = ranks.nsmallest(k).index  # most negative deviations
        overvalued = ranks.nlargest(k).index
        weights = pd.Series(0.0, index=etf_tickers)
        for iso in undervalued:
            weights[ETF_MAP[iso]] += 1.0 / k
        for iso in overvalued:
            weights[ETF_MAP[iso]] -= 1.0 / k
        weights_at_release.append((d, weights))

    if not weights_at_release:
        return mark_failed("C18_big_mac_ppp", "no valid release-date rankings")

    weight_dates = [w[0] for w in weights_at_release]
    weight_df = pd.DataFrame([w[1].values for w in weights_at_release],
                             index=pd.to_datetime(weight_dates),
                             columns=etf_tickers)

    # Reindex daily; only apply once ETF data exists. Forward-fill until next release date.
    weight_df = weight_df.sort_index()
    weight_d = weight_df.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (weight_d * rets).sum(axis=1).dropna()

    bench = rets.mean(axis=1).reindex(pnl.index)  # avg FX basket vs USD
    m = compute_metrics(pnl, benchmark=bench, name="C18 Big Mac PPP (6-mo rebal)")
    print_metrics(m)
    save_result("C18_big_mac_ppp", m, extra={
        "status": "ok",
        "rule": "Every Big Mac release: rank 6 G10 FX by USD_raw deviation. Long top-1/3 undervalued, short top-1/3 overvalued. Equal weight in each leg. Hold until next release.",
        "universe": "FXE, FXY, FXB, FXF, FXC, FXA (EUZ, JPN, GBR, CHE, CAN, AUS)",
        "source": "The Economist Big Mac index; Cumby (1996) NBER 'Forecasting Exchange Rates and Relative Prices with the Hamburger Standard'",
        "caveat": "Only 6 currencies have free yfinance ETFs; original idea uses 30+ currencies.",
    })


if __name__ == "__main__":
    main()
