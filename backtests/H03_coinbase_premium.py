"""
H03 Coinbase Premium Index.

Premium = (Coinbase BTC-USD close - OKX BTC-USDT close) / OKX BTC-USDT close.
(Binance is geo-blocked from this host, so OKX BTC-USDT is the offshore reference.)

Daily series. Compute trailing 24h (i.e. previous-day) value.
Signal: 3-day MA > +0.05% sustained -> long BTC; 3-day MA < -0.05% sustained -> flat.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import numpy as np
import pandas as pd

from harness import (
    compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


def fetch_coinbase_daily():
    cache = DATA / "coinbase_btcusd_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # Coinbase candles capped at 300 per call. Walk forward in 250-day windows.
    start = pd.Timestamp("2016-01-01")
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    rows = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=250), end)
        r = requests.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={"granularity": 86400,
                    "start": cur.isoformat() + "Z",
                    "end": nxt.isoformat() + "Z"},
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            time.sleep(1.0); continue
        for row in r.json():
            # [time, low, high, open, close, volume]
            rows.append((row[0], row[4]))
        cur = nxt
        time.sleep(0.4)
    df = pd.DataFrame(rows, columns=["ts", "close"]).drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("date").sort_index()[["close"]].rename(columns={"close": "coinbase"})
    df.to_parquet(cache)
    return df


def fetch_okx_daily():
    cache = DATA / "okx_btcusdt_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    oldest = int(pd.Timestamp.utcnow().tz_localize(None).timestamp() * 1000)
    while True:
        r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                         params={"instId": "BTC-USDT", "bar": "1D",
                                 "limit": 300, "after": oldest}, timeout=30)
        if r.status_code != 200:
            time.sleep(1.0); continue
        d = r.json()
        data = d.get("data", [])
        if not data:
            break
        for row in data:
            ts = int(row[0]); close = float(row[4])
            rows.append((ts, close))
        new_oldest = int(data[-1][0])
        if new_oldest == oldest:
            break
        oldest = new_oldest
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["ts", "close"]).drop_duplicates("ts").sort_values("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date")[["close"]].rename(columns={"close": "okx"})
    df.to_parquet(cache)
    return df


def main():
    try:
        cb = fetch_coinbase_daily()
        ok = fetch_okx_daily()
    except Exception as e:
        return mark_failed("H03_coinbase_premium", f"feed fetch failed: {e}")
    if cb.empty or ok.empty:
        return mark_failed("H03_coinbase_premium", "missing one of the spot feeds")

    cb = cb.copy(); ok = ok.copy()
    cb.index = cb.index.normalize()
    # OKX candles use a 16:00 UTC start (HK midnight). The CLOSE of a 1-day OKX
    # candle starting at D 16:00 UTC is at D+1 16:00 UTC, which is just before
    # the Coinbase D+1 23:59 UTC close. Map OKX timestamps to D+1 so we
    # compare Coinbase and OKX closes that sit within ~8 hours of each other.
    ok.index = (ok.index + pd.Timedelta(days=1)).normalize()

    df = cb.join(ok, how="inner").dropna()
    premium = (df["coinbase"] - df["okx"]) / df["okx"]

    prem_ma = premium.rolling(3).mean()

    # state machine
    state = 0
    pos = pd.Series(0.0, index=premium.index)
    for d, v in prem_ma.items():
        if pd.isna(v):
            pos.loc[d] = state
            continue
        if v > 0.0005:
            state = 1
        elif v < -0.0005:
            state = 0
        pos.loc[d] = state

    # PnL on Coinbase BTC-USD (the proxy our long actually targets)
    rets = df["coinbase"].pct_change().dropna()
    pnl = (pos.shift(1) * rets).dropna()

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index],
                        name="H03 Coinbase premium")
    print_metrics(m)
    save_result("H03_coinbase_premium", m, extra={
        "status": "ok",
        "rule": "3-day MA of (Coinbase-OKX)/OKX > +0.05% -> long BTC; < -0.05% -> flat.",
        "data_source": "Coinbase Exchange BTC-USD candles + OKX BTC-USDT candles (Binance is geo-blocked from this host).",
        "premium_mean_bps": float(premium.mean() * 10000),
        "premium_std_bps": float(premium.std() * 10000),
        "n_long_days": int((pos == 1).sum()),
    })


if __name__ == "__main__":
    main()
