"""
Shared OHLCV loader (single ticker) with parquet cache.
Returns a DataFrame with columns: open, high, low, close, volume.
auto_adjust=True so all OHLC are adjusted.
"""
import warnings
import datetime as dt
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def load_ohlcv(ticker, start="2000-01-01", end=None, cache=True):
    import yfinance as yf
    end = end or dt.date.today().isoformat()
    fp = DATA / f"ohlcv_{ticker}_{start}_{end}.parquet"
    if cache and fp.exists():
        return pd.read_parquet(fp)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna(how="any").sort_index()
    if cache:
        df.to_parquet(fp)
    return df


def atr(ohlcv, n=14):
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
