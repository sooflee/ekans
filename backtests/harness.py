"""
Shared backtest utilities for the signals catalog.

Conventions:
- All backtests produce daily PnL series in `pnl` (excess returns, decimal).
- Metrics computed: CAGR, Sharpe (252-day), max drawdown, Calmar, hit rate, t-stat.
- Benchmark = SPY buy-and-hold over the same window.
- All data pulled from yfinance / FRED / public APIs. Cached to data/.

Run a backtest file directly:
    .venv/bin/python backtests/signal_XX_name.py
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def save_result(signal_id, metrics, extra=None):
    """Persist a signal's backtest metrics to results/<signal_id>.json.

    metrics: dict from compute_metrics()
    extra: optional dict of extra fields (e.g., rule, source, status='ok'|'fail')
    """
    payload = dict(metrics)
    payload["signal_id"] = signal_id
    if extra:
        payload.update(extra)
    fp = RESULTS / f"{signal_id}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return fp


def mark_failed(signal_id, reason, extra=None):
    """Record a signal that couldn't be backtested."""
    payload = {"signal_id": signal_id, "status": "fail", "reason": reason}
    if extra:
        payload.update(extra)
    fp = RESULTS / f"{signal_id}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return fp


# ---------- data loaders ----------

def load_prices(tickers, start="2000-01-01", end=None, cache=True):
    """Load adjusted-close prices for one or more tickers from yfinance.
    Returns a DataFrame indexed by date, columns = tickers."""
    import yfinance as yf
    if isinstance(tickers, str):
        tickers = [tickers]
    end = end or dt.date.today().isoformat()
    key = "_".join(sorted(tickers)) + f"_{start}_{end}.parquet"
    fp = DATA / key
    if cache and fp.exists():
        return pd.read_parquet(fp)
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif "Close" in df.columns:
        df = df[["Close"]]
        df.columns = tickers
    df = df.dropna(how="all").sort_index()
    if cache:
        df.to_parquet(fp)
    return df


def load_fred(series, start="1990-01-01", end=None, cache=True):
    """Load one or more series from FRED."""
    from pandas_datareader import data as pdr
    if isinstance(series, str):
        series = [series]
    end = end or dt.date.today().isoformat()
    key = "fred_" + "_".join(sorted(series)) + f"_{start}_{end}.parquet"
    fp = DATA / key
    if cache and fp.exists():
        return pd.read_parquet(fp)
    df = pdr.DataReader(series, "fred", start=start, end=end)
    df = df.dropna(how="all").sort_index()
    if cache:
        df.to_parquet(fp)
    return df


# ---------- metrics ----------

def compute_metrics(pnl, benchmark=None, name="Strategy"):
    """Given a daily PnL/return series, compute summary metrics.
    pnl: pd.Series of daily decimal returns (excess or total, your choice; be consistent).
    Returns dict.
    """
    pnl = pnl.dropna()
    if len(pnl) < 30:
        return {"name": name, "n_days": len(pnl), "error": "insufficient data"}

    ann_factor = 252
    eq = (1 + pnl).cumprod()
    years = len(pnl) / ann_factor
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl.std() * np.sqrt(ann_factor)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(ann_factor) if pnl.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    hit = (pnl > 0).mean()
    # t-stat for mean return ≠ 0
    t_stat = pnl.mean() / (pnl.std() / np.sqrt(len(pnl))) if pnl.std() > 0 else 0

    out = {
        "name": name,
        "start": str(pnl.index[0].date()),
        "end": str(pnl.index[-1].date()),
        "n_days": len(pnl),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar) if not np.isnan(calmar) else None,
        "hit_rate": float(hit),
        "t_stat": float(t_stat),
    }

    if benchmark is not None:
        b = benchmark.reindex(pnl.index).dropna()
        if len(b) > 30:
            b_eq = (1 + b).cumprod()
            b_years = len(b) / ann_factor
            out["bench_cagr"] = float(b_eq.iloc[-1] ** (1 / b_years) - 1)
            out["bench_sharpe"] = float(b.mean() / b.std() * np.sqrt(ann_factor)) if b.std() > 0 else 0
            out["excess_cagr"] = out["cagr"] - out["bench_cagr"]

    return out


def print_metrics(m):
    """Pretty-print a metrics dict."""
    print(f"--- {m.get('name','?')} ---")
    if "error" in m:
        print(f"  error: {m['error']}")
        return
    print(f"  period:    {m['start']} → {m['end']} ({m['n_days']} days)")
    print(f"  CAGR:      {m['cagr']*100:7.2f}%")
    print(f"  Ann.Vol:   {m['ann_vol']*100:7.2f}%")
    print(f"  Sharpe:    {m['sharpe']:7.2f}")
    print(f"  MaxDD:     {m['max_dd']*100:7.2f}%")
    if m.get('calmar') is not None:
        print(f"  Calmar:    {m['calmar']:7.2f}")
    print(f"  HitRate:   {m['hit_rate']*100:7.2f}%")
    print(f"  t-stat:    {m['t_stat']:7.2f}")
    if "bench_cagr" in m:
        print(f"  Bench:     CAGR {m['bench_cagr']*100:.2f}%  Sharpe {m['bench_sharpe']:.2f}")
        print(f"  Excess CAGR: {m['excess_cagr']*100:.2f}%")


def daily_returns(prices):
    """Convert price series/frame to daily simple returns."""
    return prices.pct_change().dropna(how="all")


# ---------- a couple of common building blocks ----------

def long_short_pnl(positions, returns):
    """Positions in {-1, 0, +1} or floats. Returns is same-shape returns frame.
    Position is applied to NEXT day's return (no look-ahead)."""
    pos = positions.shift(1)
    if isinstance(pos, pd.DataFrame):
        return (pos * returns).sum(axis=1)
    return pos * returns


def rolling_zscore(s, window):
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


if __name__ == "__main__":
    # smoke test
    px = load_prices(["SPY", "TLT"], start="2010-01-01")
    print("Loaded:", px.shape, "tickers:", list(px.columns))
    r = daily_returns(px)
    m = compute_metrics(r["SPY"], benchmark=r["SPY"], name="SPY buy-and-hold")
    print_metrics(m)
