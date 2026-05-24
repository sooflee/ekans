"""
R-D11 Short interest (retry).

Original spec: rank by bi-monthly FINRA short interest as %-of-float;
long bottom decile, short top decile, monthly rebalance.

FINRA CDN is gated (403). However, FINRA's open API at
https://api.finra.org/data/group/otcMarket/name/regShoDaily exposes
*daily* short-volume by symbol (~1 year rolling). This is a known proxy
for short interest (Diether et al. 2009 -- 'Short-sale strategies and
return predictability'). We use the daily short-volume ratio (SVR =
shortParQuantity / totalParQuantity) averaged over a rolling 21d window
as the cross-sectional ranking signal.

Universe: ~30 liquid mid/large caps with full 1y coverage.
"""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save_result, mark_failed, compute_metrics, load_prices, DATA


SIGNAL_ID = "R-D11_short_interest"

# 30 liquid mid/large caps spanning multiple sectors
UNIVERSE = [
    # mega: a few for baseline liquidity
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    # mid/large diverse
    "JPM", "BAC", "C", "WFC",
    "XOM", "CVX", "OXY", "SLB",
    "KO", "PEP", "MCD", "SBUX",
    "GE", "BA", "CAT", "DE",
    "T", "VZ",
    "WMT", "TGT", "COST",
    "F", "GM",
]


def fetch_short_volume(symbol):
    """Fetch ~1y of daily short-volume rows for a symbol from FINRA API."""
    fp = DATA / f"finra_regsho_{symbol}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
    payload = {
        "compareFilters": [
            {"fieldName": "securitiesInformationProcessorSymbolIdentifier",
             "fieldValue": symbol, "compareType": "EQUAL"}
        ],
        "limit": 5000,
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200 or len(r.text) < 100:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return None
    # Aggregate across reporting facilities for each (date, symbol)
    df = df.groupby("tradeReportDate", as_index=False).agg({
        "shortParQuantity": "sum",
        "totalParQuantity": "sum",
    })
    df["svr"] = df["shortParQuantity"] / df["totalParQuantity"]
    df["date"] = pd.to_datetime(df["tradeReportDate"])
    df = df.set_index("date").sort_index()[["svr"]]
    df.to_parquet(fp)
    return df


def main():
    # 1) Fetch SVR for each symbol
    svr_frames = {}
    print("Fetching FINRA Reg SHO Daily ...")
    for s in UNIVERSE:
        try:
            df = fetch_short_volume(s)
            if df is None or len(df) < 100:
                print(f"  {s}: skipped (insufficient FINRA data)")
                continue
            svr_frames[s] = df["svr"]
        except Exception as e:
            print(f"  {s}: {e}")
    if len(svr_frames) < 10:
        return mark_failed(SIGNAL_ID, f"only {len(svr_frames)} symbols had FINRA data")

    svr = pd.DataFrame(svr_frames).sort_index()
    svr = svr.dropna(how="all")
    print(f"  SVR matrix: {svr.shape}, {svr.index[0].date()} to {svr.index[-1].date()}")

    # 2) Smooth with rolling 21d mean
    svr_smooth = svr.rolling(21, min_periods=10).mean()

    # 3) Get prices for the universe over the same period
    start = svr.index[0].date().isoformat()
    end = (svr.index[-1].date() + pd.Timedelta(days=1)).isoformat()
    px = load_prices(list(svr.columns), start=start, end=end)
    px = px.reindex(columns=list(svr.columns)).dropna(how="all")
    ret = px.pct_change()
    print(f"  prices: {px.shape}, returns: {ret.shape}")

    # 4) Monthly rebalance: at each month-end, rank by SVR smooth -> bottom decile long, top decile short
    # With 30 names, top/bottom decile ~ 3 names each. Use top 5 / bottom 5 to keep more diversification.
    n = svr_smooth.shape[1]
    k = max(3, n // 6)  # ~5 names per side (sextiles)
    print(f"  basket size each side: {k}")

    # Reindex SVR to trading days
    svr_smooth_d = svr_smooth.reindex(ret.index).ffill()

    # Generate monthly rebalance dates (month-end trading day)
    month_ends = ret.resample("M").apply(lambda x: x.index[-1] if len(x) else None).iloc[:, 0]
    month_ends = month_ends.dropna()
    rebal_dates = pd.DatetimeIndex(month_ends.values).intersection(ret.index)
    print(f"  rebalance dates: {len(rebal_dates)}")

    # Build positions: at each rebal_date, set positions for the next month
    pos = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    holding = pd.Series(0.0, index=ret.columns)

    for d in ret.index:
        if d in rebal_dates:
            row = svr_smooth_d.loc[d].dropna()
            if len(row) < 2 * k:
                continue
            ranks = row.sort_values()
            longs = ranks.index[:k]  # bottom SVR -> long
            shorts = ranks.index[-k:]  # top SVR -> short
            holding[:] = 0.0
            holding[longs] = 1.0 / k
            holding[shorts] = -1.0 / k
        pos.loc[d] = holding

    # Apply positions to next-day returns
    pos = pos.shift(1).fillna(0.0)
    pnl = (pos * ret).sum(axis=1).dropna()

    # Benchmark: equal-weighted long-only of the universe
    bench = ret.mean(axis=1)

    metrics = compute_metrics(pnl, benchmark=bench,
                              name="FINRA SVR low-minus-high (sextile, monthly)")
    print("Metrics:", metrics)

    # Count rebal events that produced a position
    n_rebals = int(((pos.diff().abs().sum(axis=1) > 0)).sum())

    extra = {
        "rule": "Monthly rebalance: long bottom-sextile / short top-sextile by 21d-smoothed daily short-volume ratio (FINRA Reg SHO Daily).",
        "mechanism": "Heavily-shorted stocks underperform; constrained short supply makes informed shorts profitable (Diether/Lee/Werner 2009).",
        "source": "FINRA api.finra.org regShoDaily (daily short volume); yfinance OHLC.",
        "n_events": n_rebals,
        "universe": UNIVERSE,
        "basket_size_per_side": k,
        "data_substitution": "FINRA bi-monthly short-interest bulk CDN gated (403). Substituted daily short-volume ratio (SVR) from FINRA's open Reg SHO Daily API. SVR is a known SI proxy (Diether/Lee/Werner 2009). History limited to ~1y rolling window.",
        "status": "ok",
    }
    save_result(SIGNAL_ID, metrics, extra=extra)
    return metrics


if __name__ == "__main__":
    main()
