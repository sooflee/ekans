"""
H02 Stablecoin Supply Ratio (SSR) Bollinger oscillator.

SSR = BTC market cap / aggregate stablecoin supply.
  - BTC mcap = BTC-USD price (yfinance) * circulating supply
    (proxy: blockchain.info 'total-bitcoins' chart).
  - Stablecoin total supply from DefiLlama stablecoincharts/all
    (aggregating peggedUSD totalCirculatingUSD; daily, from 2017-12).
Bollinger oscillator = (SSR - SMA200) / (2 * SD200). Range ~[-1, +1].
Rule: long BTC when oscillator < -0.80 (stablecoin supply dwarfs BTC mcap
relative to the trend -> dry powder accumulation). Flat otherwise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


def fetch_stablecoin_supply():
    cache = DATA / "defillama_stablecoins_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=60)
    j = r.json()
    rows = []
    for d in j:
        ts = int(d["date"])
        usd = (d.get("totalCirculatingUSD", {}) or {}).get("peggedUSD", 0) or 0
        rows.append((pd.Timestamp(ts, unit="s"), float(usd)))
    df = pd.DataFrame(rows, columns=["date", "stable_usd"]).set_index("date").sort_index()
    df.to_parquet(cache)
    return df


def fetch_btc_supply():
    cache = DATA / "blockchain_info_total_bitcoins.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get("https://api.blockchain.info/charts/total-bitcoins?timespan=all&format=json", timeout=60)
    j = r.json()
    rows = [(pd.Timestamp(v["x"], unit="s"), float(v["y"])) for v in j["values"]]
    df = pd.DataFrame(rows, columns=["date", "supply"]).set_index("date").sort_index()
    df.to_parquet(cache)
    return df


def main():
    try:
        stable = fetch_stablecoin_supply()["stable_usd"]
        supply = fetch_btc_supply()["supply"]
    except Exception as e:
        return mark_failed("H02_stablecoin_supply_ratio", f"data fetch failed: {e}")

    px = load_prices(["BTC-USD"], start="2014-01-01").iloc[:, 0]

    # daily grid, forward fill
    idx = pd.date_range(start=max(stable.index.min(), supply.index.min(), px.index.min()),
                        end=min(stable.index.max(), supply.index.max(), px.index.max()),
                        freq="D")
    stable_d = stable.resample("D").last().ffill().reindex(idx).ffill()
    supply_d = supply.resample("D").last().ffill().reindex(idx).ffill()
    px_d = px.reindex(idx).ffill()

    btc_mcap = px_d * supply_d
    ssr = btc_mcap / stable_d
    ssr = ssr.replace([np.inf, -np.inf], np.nan).dropna()

    sma = ssr.rolling(200).mean()
    sd = ssr.rolling(200).std()
    osc = (ssr - sma) / (2 * sd)

    pos = (osc < -0.80).astype(float)

    rets = px_d.pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[osc.dropna().index[0]:]

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index],
                        name="H02 SSR Bollinger oscillator")
    print_metrics(m)
    save_result("H02_stablecoin_supply_ratio", m, extra={
        "status": "ok",
        "rule": "Long BTC when (SSR - SMA200)/(2*SD200) < -0.80; flat otherwise.",
        "data_source": "DefiLlama stablecoincharts/all (peggedUSD USD-circulating); BTC supply from blockchain.info 'total-bitcoins'; BTC price from yfinance.",
        "n_long_days": int((pos == 1).sum()),
        "n_total_days": int(len(pos)),
    })


if __name__ == "__main__":
    main()
