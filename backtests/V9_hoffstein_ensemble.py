"""
V9 Hoffstein 8-model trend ensemble (STRONG CANDIDATE)
8 trend models on SPY (daily prices). Each casts one vote (long=1, else 0):
  1. SMA50 > SMA200 (golden-cross state)
  2. 12-month return > 0 (excluding most recent month)
  3. 3-month rate-of-change > 0
  4. Price > 50-day Donchian high (rolling 50d max of prior close)
  5. MACD line (EMA12 - EMA26) > Signal (EMA9 of MACD)
  6. 100-day linear-regression slope on price > 0
  7. Price > KAMA(10,2,30) adaptive MA
  8. BB(20,2) breakout: close > upper Bollinger band (price > 20d MA + 2 std)
Exposure to SPY = votes / 8 (between 0 and 1). Rest in cash (0%).
Mechanism (Corey Hoffstein / Newfound): "diversifying the model" reduces
specification risk in trend-following — any single trend rule has timing luck,
but an ensemble averages it out and produces a smoother equity curve with
similar return and lower drawdown vs SPY B&H.
Backtest 1990+ (SPY since 1993).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def ema(x, span):
    return x.ewm(span=span, adjust=False).mean()


def kama(price, er_period=10, fast=2, slow=30):
    change = (price - price.shift(er_period)).abs()
    vol = price.diff().abs().rolling(er_period).sum()
    er = change / vol.replace(0, np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = sc.fillna(0.0)
    out = pd.Series(np.nan, index=price.index)
    out.iloc[er_period] = price.iloc[er_period]
    for i in range(er_period + 1, len(price)):
        prev = out.iloc[i-1]
        if np.isnan(prev):
            prev = price.iloc[i-1]
        out.iloc[i] = prev + sc.iloc[i] * (price.iloc[i] - prev)
    return out


def regression_slope(price, window=100):
    """Slope of OLS regression of log price on time, sign-only is what we need."""
    logp = np.log(price)
    x = np.arange(window)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = (x_dev ** 2).sum()
    def slope_fn(arr):
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        return ((arr - y_mean) * x_dev).sum() / denom
    return logp.rolling(window).apply(slope_fn, raw=True)


def main():
    try:
        spy = load_prices(["SPY"], start="1993-01-29").iloc[:, 0]
    except Exception as e:
        return mark_failed("V9_hoffstein_ensemble", f"data load: {e}")

    px = spy.dropna()
    rets = px.pct_change()

    # Model 1: SMA50 > SMA200
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    v1 = (sma50 > sma200).astype(float)

    # Model 2: 12m return > 0 excluding most recent month
    #          = (price.shift(21) / price.shift(252)) - 1 > 0
    v2 = ((px.shift(21) / px.shift(252)) - 1 > 0).astype(float)

    # Model 3: 3-month ROC > 0
    v3 = (px.pct_change(63) > 0).astype(float)

    # Model 4: Donchian-50 breakout — price > 50d prior high
    don50 = px.shift(1).rolling(50).max()
    v4 = (px > don50).astype(float)

    # Model 5: MACD > signal
    macd = ema(px, 12) - ema(px, 26)
    sig = ema(macd, 9)
    v5 = (macd > sig).astype(float)

    # Model 6: 100d regression slope > 0
    slope = regression_slope(px, 100)
    v6 = (slope > 0).astype(float)

    # Model 7: price > KAMA
    k = kama(px, 10, 2, 30)
    v7 = (px > k).astype(float)

    # Model 8: BB(20,2) breakout — close > 20d MA + 2 std
    ma20 = px.rolling(20).mean()
    sd20 = px.rolling(20).std()
    upper = ma20 + 2 * sd20
    v8 = (px > upper).astype(float)

    votes = pd.concat([v1, v2, v3, v4, v5, v6, v7, v8], axis=1)
    votes.columns = [f"m{i+1}" for i in range(8)]
    exposure = votes.sum(axis=1) / 8.0

    pnl = (exposure.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.index >= "1995-01-01"]  # warm-up
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V9 Hoffstein 8-model trend ensemble")
    print_metrics(m)
    save_result("V9_hoffstein_ensemble", m, extra={
        "status": "ok",
        "rule": "8 trend models on SPY (SMA50>SMA200, 12m>0 ex-last-month, 3m ROC>0, Donchian50, MACD>signal, 100d slope>0, price>KAMA, BB breakout). Exposure = votes/8.",
        "mechanism": "Diversifying the trend model reduces specification timing-luck risk; ensemble of correlated-but-different signals smooths the equity curve.",
        "source": "Corey Hoffstein (Newfound Research), YouTube interview round 2 (Phase 1V).",
        "models": ["SMA50>SMA200","12mROC ex-1m>0","3mROC>0","Donchian50","MACD>signal","100d-slope>0","price>KAMA","BB(20,2) breakout"],
    })


if __name__ == "__main__":
    main()
