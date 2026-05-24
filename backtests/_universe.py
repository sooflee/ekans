"""
Fixed basket of ~50 liquid US large-cap tickers with deep history (mostly listed
pre-2005). Used by D03, D05, D06, D09, D14, D17 as a simplified equity universe.

This is a STATIC, look-ahead-biased universe: every name here survived to today
and is large enough to be liquid. Therefore results materially over-state
realisable returns vs a CRSP point-in-time universe. Treat absolute Sharpe with
skepticism; relative ranking across factors is the useful comparison.
"""

UNIVERSE = sorted(set([
    # Mega-cap tech / comms
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "IBM",
    "INTC", "ADBE", "QCOM", "TXN", "AMAT",
    # Communications / media
    "T", "VZ", "CMCSA", "DIS", "NFLX",
    # Consumer disc
    "HD", "LOW", "NKE", "MCD", "SBUX", "TGT", "COST",
    # Consumer staples
    "WMT", "PG", "KO", "PEP", "MO", "CL", "KMB",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABT", "LLY", "UNH", "BMY", "GILD", "AMGN", "MDT",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "BLK", "SCHW",
    # Industrials
    "GE", "BA", "CAT", "HON", "MMM", "UNP", "LMT", "RTX", "UPS", "DE",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Materials
    "DD", "DOW", "FCX", "NEM",
    # Utilities / Real estate
    "NEE", "DUK", "SO", "AMT",
]))

# Reasonable start date — most names have full coverage from here
DEFAULT_START = "2002-01-01"


def load_universe_prices(start=DEFAULT_START, end=None, include_spy=True):
    """Load adjusted-close prices for the basket, with a hashed cache file
    (the default harness loader uses ticker-list-as-filename which exceeds
    macOS's 255-char NAME_MAX for >50 tickers)."""
    import datetime as dt
    import hashlib
    from pathlib import Path
    import pandas as pd
    import yfinance as yf

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    DATA.mkdir(exist_ok=True)

    tickers = sorted(set(UNIVERSE + (["SPY"] if include_spy else [])))
    end = end or dt.date.today().isoformat()
    key_src = ",".join(tickers) + f"|{start}|{end}"
    h = hashlib.sha1(key_src.encode()).hexdigest()[:10]
    fp = DATA / f"universe_{len(tickers)}_{start}_{end}_{h}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    df = df.dropna(how="all").sort_index()
    df.to_parquet(fp)
    return df
