"""
R-L14 McClellan oscillator (retry).

Original spec: NYSE McClellan = 19d EMA(Adv-Dec) - 39d EMA(Adv-Dec).
NYSE breadth feeds are gated. Substitution: compute breadth from S&P 500
constituents (same approach as R-L13) and call it SP500-McClellan.

Rule: When SP500-McClellan crosses upward through zero from below -50,
go long SPY for next 21 sessions (~one month).
"""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save_result, mark_failed, compute_metrics, load_prices, DATA


SIGNAL_ID = "R-L14_mcclellan"


def get_sp500_tickers():
    fp = DATA / "sp500_components.parquet"
    if fp.exists():
        return pd.read_parquet(fp)["Symbol"].tolist()
    r = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    df = pd.read_html(io.StringIO(r.text))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df[["Symbol"]].to_parquet(fp)
    return df["Symbol"].tolist()


def download_ohlcv(tickers, start, end):
    fp = DATA / f"sp500_ohlcv_{start}_{end}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = yf.download(tickers, start=start, end=end, progress=False,
                     auto_adjust=True, group_by="ticker", threads=True)
    df.to_parquet(fp)
    return df


def build_adv_dec(ohlcv, tickers):
    close = {}
    for t in tickers:
        try:
            sub = ohlcv[t]
            if isinstance(sub, pd.DataFrame):
                close[t] = sub["Close"]
        except Exception:
            continue
    close = pd.DataFrame(close).dropna(how="all", axis=1)
    ret = close.pct_change()
    adv_n = (ret > 0).sum(axis=1)
    dec_n = (ret < 0).sum(axis=1)
    return (adv_n - dec_n).rename("AD").dropna()


def main():
    start = "2014-01-01"
    end = "2026-05-22"
    tickers = get_sp500_tickers()
    print(f"S&P 500 components: {len(tickers)}")
    try:
        ohlcv = download_ohlcv(tickers, start, end)
    except Exception as e:
        return mark_failed(SIGNAL_ID, f"bulk download failed: {e}")
    print(f"  ohlcv shape: {ohlcv.shape}")

    ad = build_adv_dec(ohlcv, tickers)
    print(f"  Adv-Dec: {len(ad)} days, mean={ad.mean():.1f}")

    # McClellan oscillator = EMA19(AD) - EMA39(AD)
    ema19 = ad.ewm(span=19, adjust=False).mean()
    ema39 = ad.ewm(span=39, adjust=False).mean()
    mco = (ema19 - ema39).dropna()
    print(f"  McClellan: range [{mco.min():.1f}, {mco.max():.1f}], std={mco.std():.1f}")

    if len(mco) < 200:
        return mark_failed(SIGNAL_ID, "insufficient McClellan history")

    spy = load_prices(["SPY"], start=start, end=end)
    spy_ret = spy["SPY"].pct_change()

    # Threshold scaled to our oscillator's std. Original rule uses -50 vs NYSE McClellan std ~100;
    # our SP500-based oscillator std ~20, so use -2.5 * sigma (i.e., scale -50 by std/100).
    std = mco.std()
    scaled_oversold = -2.5 * std  # mirrors -50 / 20 ~ -2.5 sigma for NYSE TRIN
    # Signal: prev value < scaled_oversold and current >= 0 -> upward cross through zero from oversold
    prev = mco.shift(1)
    # require the lowest reading in the last 5 days to have been below threshold (not just yesterday)
    recent_min = mco.rolling(5).min().shift(1)
    signal = (recent_min < scaled_oversold) & (prev < 0) & (mco >= 0)
    n_events = int(signal.sum())
    print(f"  scaled-oversold threshold: {scaled_oversold:.1f}")
    print(f"  cross-up-from-oversold events: {n_events}")

    pos = pd.Series(0.0, index=mco.index)
    for d in signal[signal].index:
        loc = pos.index.get_loc(d)
        for k in range(1, 22):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    pos = pos.reindex(spy_ret.index).fillna(0.0)
    pnl = pos * spy_ret

    metrics = compute_metrics(pnl.dropna(), benchmark=spy_ret, name="SP500-McClellan cross-up long SPY 21d")
    print("Metrics:", metrics)

    extra = {
        "rule": "When SP500-McClellan crosses up through 0 within 5 days of a sub --2.5sigma reading, long SPY next 21 sessions.",
        "mechanism": "Breadth thrust off an oversold reading historically precedes multi-week SPX rallies (Zweig breadth thrust analog).",
        "source": "S&P 500 components from Wikipedia + yfinance OHLCV; oscillator = EMA19(Adv-Dec) - EMA39(Adv-Dec).",
        "n_events": n_events,
        "data_substitution": "Original NYSE breadth (^ADV/^DECL) unavailable; substituted SP500 constituent advance-decline.",
        "status": "ok",
    }
    save_result(SIGNAL_ID, metrics, extra=extra)
    return metrics


if __name__ == "__main__":
    main()
