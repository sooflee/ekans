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


def save_result(signal_id, metrics, extra=None, pnl=None):
    """Persist a signal's backtest metrics to results/<signal_id>.json.

    metrics: dict from compute_metrics()
    extra: optional dict of extra fields (e.g., rule, source, status='ok'|'fail')
    pnl: optional pd.Series of daily PnL — saved to results/pnl/ for correlation analysis.
    """
    payload = dict(metrics)
    payload["signal_id"] = signal_id
    if extra:
        payload.update(extra)
    fp = RESULTS / f"{signal_id}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if pnl is not None:
        pnl_dir = RESULTS / "pnl"
        pnl_dir.mkdir(exist_ok=True)
        pnl.to_frame("pnl").to_parquet(pnl_dir / f"{signal_id}.parquet")
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
    """Load one or more series from FRED. Uses fredapi if available (more reliable), falls back to pandas_datareader."""
    if isinstance(series, str):
        series = [series]
    end = end or dt.date.today().isoformat()
    key = "fred_" + "_".join(sorted(series)) + f"_{start}_{end}.parquet"
    fp = DATA / key
    if cache and fp.exists():
        return pd.read_parquet(fp)
    try:
        from fredapi import Fred
        import os
        api_key = os.environ.get("FRED_API_KEY", "032756edd8cb0e81fc95bb23aeefc18a")
        fred = Fred(api_key=api_key)
        frames = {}
        for s in series:
            frames[s] = fred.get_series(s, observation_start=start, observation_end=end)
        df = pd.DataFrame(frames)
    except ImportError:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(series, "fred", start=start, end=end)
    df = df.dropna(how="all").sort_index()
    if cache:
        df.to_parquet(fp)
    return df


# ---------- metrics ----------

def compute_metrics(pnl, benchmark=None, name="Strategy", positions=None, cost_bps=10):
    """Given a daily PnL/return series, compute summary metrics.
    pnl: pd.Series of daily decimal returns (excess or total, your choice; be consistent).
    positions: optional pd.Series of positions (e.g. 1/0/-1) for transaction cost calc.
               Costs are deducted on each day the position changes, proportional to |delta|.
    cost_bps: round-trip cost in basis points (default 10 = 5bps each way).
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

    # --- Transaction cost adjustment ---
    if positions is not None:
        pos = positions.reindex(pnl.index).fillna(0)
        turnover = pos.diff().abs().fillna(0)
        daily_cost = turnover * (cost_bps / 10000)
        net_pnl = pnl - daily_cost
        net_pnl = net_pnl.dropna()
        if len(net_pnl) > 30:
            net_eq = (1 + net_pnl).cumprod()
            net_years = len(net_pnl) / ann_factor
            net_cagr = net_eq.iloc[-1] ** (1 / net_years) - 1
            net_sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(ann_factor) if net_pnl.std() > 0 else 0
            net_dd = (net_eq / net_eq.cummax() - 1)
            out["net_cagr"] = float(net_cagr)
            out["net_sharpe"] = float(net_sharpe)
            out["net_max_dd"] = float(net_dd.min())
            out["turnover_annual"] = float(turnover.sum() / years)
            out["cost_drag"] = float(cagr - net_cagr)
            out["cost_bps"] = cost_bps

    # --- In-sample / Out-of-sample split ---
    mid = len(pnl) // 2
    if mid >= 60:
        for label, chunk in [("is", pnl.iloc[:mid]), ("oos", pnl.iloc[mid:])]:
            c_eq = (1 + chunk).cumprod()
            c_years = len(chunk) / ann_factor
            c_cagr = c_eq.iloc[-1] ** (1 / c_years) - 1
            c_sharpe = chunk.mean() / chunk.std() * np.sqrt(ann_factor) if chunk.std() > 0 else 0
            out[f"{label}_cagr"] = float(c_cagr)
            out[f"{label}_sharpe"] = float(c_sharpe)
            out[f"{label}_start"] = str(chunk.index[0].date())
            out[f"{label}_end"] = str(chunk.index[-1].date())

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
