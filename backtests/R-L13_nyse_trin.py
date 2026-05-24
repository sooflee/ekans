"""
R-L13 NYSE TRIN (retry).

Original spec called for NYSE TRIN = (Adv/Dec)/(AdvVol/DecVol) from Stooq/Yahoo.
Both are gated. Substitution: reconstruct a TRIN-equivalent using the S&P 500
component universe (constituent list from Wikipedia, prices/volumes from
yfinance). This is "SP500-TRIN", a high-fidelity proxy for the genuine
NYSE-TRIN since SP500 dominates NYSE composite turnover.

Rule: When SP500-TRIN close > 2.5, go long SPY for next 5 sessions.
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
from harness import save_result, mark_failed, compute_metrics, load_prices, daily_returns, DATA


SIGNAL_ID = "R-L13_nyse_trin"


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
    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    df.to_parquet(fp)
    return df


def build_trin(ohlcv, tickers):
    """Compute daily SP500-TRIN.
    TRIN = (Advancers/Decliners) / (AdvVolume/DecVolume).
    """
    # Build per-ticker close & volume frames
    close = {}
    vol = {}
    for t in tickers:
        try:
            sub = ohlcv[t]
            if isinstance(sub, pd.DataFrame):
                close[t] = sub["Close"]
                vol[t] = sub["Volume"]
        except Exception:
            continue
    close = pd.DataFrame(close)
    vol = pd.DataFrame(vol)
    close = close.dropna(how="all", axis=1)
    vol = vol.reindex(columns=close.columns)

    ret = close.pct_change()
    # Advancers/decliners: by sign of daily return
    adv = (ret > 0).astype(int)
    dec = (ret < 0).astype(int)
    # For volume side, use the ticker's day-volume only if it advanced/declined
    adv_vol = vol.where(ret > 0, 0)
    dec_vol = vol.where(ret < 0, 0)

    adv_n = adv.sum(axis=1)
    dec_n = dec.sum(axis=1)
    adv_v = adv_vol.sum(axis=1)
    dec_v = dec_vol.sum(axis=1)

    # Avoid div/zero
    ratio_issues = adv_n / dec_n.replace(0, np.nan)
    ratio_vol = adv_v / dec_v.replace(0, np.nan)
    trin = ratio_issues / ratio_vol
    return trin.dropna()


def main():
    start = "2014-01-01"
    end = "2026-05-22"
    print("Fetching S&P 500 components...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} symbols")

    print("Downloading OHLCV (cached) ...")
    try:
        ohlcv = download_ohlcv(tickers, start, end)
    except Exception as e:
        return mark_failed(SIGNAL_ID, f"yfinance bulk download failed: {e}")

    print(f"  ohlcv shape: {ohlcv.shape}")
    print("Building SP500-TRIN ...")
    trin = build_trin(ohlcv, tickers)
    print(f"  TRIN: {len(trin)} days, mean={trin.mean():.2f}, median={trin.median():.2f}, p95={trin.quantile(0.95):.2f}")

    if len(trin) < 200:
        return mark_failed(SIGNAL_ID, f"insufficient TRIN history: {len(trin)} days")

    # Get SPY returns
    spy = load_prices(["SPY"], start=start, end=end)
    spy_ret = spy["SPY"].pct_change()

    # Rule: TRIN close > 2.5 -> long SPY next 5 sessions
    threshold = 2.5
    signal = trin > threshold
    n_events = int(signal.sum())
    print(f"  events (TRIN > {threshold}): {n_events}")

    # Position: when signal fires on day t, hold long days t+1..t+5
    pos = pd.Series(0.0, index=trin.index)
    sig_idx = signal[signal].index
    for d in sig_idx:
        # find next 5 trading days
        loc = pos.index.get_loc(d)
        # set positions for offsets 1..5 (avoid overlap by max)
        for k in range(1, 6):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    # Realize PnL: pos already represents "we are long today" (already shifted)
    pos = pos.reindex(spy_ret.index).fillna(0.0)
    pnl = pos * spy_ret

    metrics = compute_metrics(pnl.dropna(), benchmark=spy_ret, name="SP500-TRIN > 2.5 long SPY 5d")
    print("Metrics:", metrics)

    extra = {
        "rule": "Long SPY for next 5 sessions whenever SP500-TRIN close > 2.5 (overlap permitted).",
        "mechanism": "Extreme down-volume vs declining issues marks short-term capitulation; mean-reversion bounces over 1 week.",
        "source": "S&P 500 components from Wikipedia; OHLCV from yfinance. TRIN reconstructed from constituent up/down counts + up/down volume.",
        "n_events": n_events,
        "trin_threshold": threshold,
        "data_substitution": "Original NYSE-TRIN unavailable (Stooq apikey-gated, yfinance 404). Substituted with reconstructed SP500-TRIN from constituent universe.",
        "status": "ok",
    }
    save_result(SIGNAL_ID, metrics, extra=extra)
    return metrics


if __name__ == "__main__":
    main()
