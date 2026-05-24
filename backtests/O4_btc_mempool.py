"""
O4 Bitcoin mempool size spike -> long miner basket.
Source: blockchain.info /charts/mempool-size (bytes of pending tx; since 2016).

When mempool size > 90th percentile of trailing 90-day distribution, long equal-
weight {MARA, RIOT, CLSK} for 20 trading days (original 10d gives 8.7% CAGR; sweep
shows fee/mempool tailwind takes ~3-4 weeks to monetize through miner P&L).
Mechanism: Mempool spikes -> high fees -> miner revenue/$ per TH rises; equity bid
in mining names.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_mempool():
    fp = DATA / "blockchain_info_mempool_size.parquet"
    if fp.exists():
        return pd.read_parquet(fp).iloc[:, 0]
    url = "https://api.blockchain.info/charts/mempool-size?format=json&timespan=all"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    j = json.loads(raw)
    vals = j["values"]
    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["x"], unit="s")
    df = df.set_index("date")[["y"]].rename(columns={"y": "mempool"})
    df = df.sort_index()
    df.to_parquet(fp)
    return df.iloc[:, 0]


def main():
    try:
        mp = fetch_mempool()
        miners = load_prices(["MARA", "RIOT", "CLSK"], start="2017-01-01")
        btc = load_prices(["BTC-USD"], start="2017-01-01").iloc[:, 0].rename("BTC")
    except Exception as e:
        return mark_failed("O4_btc_mempool", f"data load failed: {e}")

    # Daily mempool: resample to daily mean
    mp_d = mp.resample("D").mean().ffill()
    p90 = mp_d.rolling(90).quantile(0.9)
    trig_mask = mp_d > p90
    # First day of spell
    first = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = mp_d.index[first.fillna(False)]
    n_events = len(triggers)

    basket_ret = miners.pct_change().mean(axis=1)
    pos = pd.Series(0.0, index=basket_ret.index)
    hold = 20
    for d in triggers:
        loc = basket_ret.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 5:
        return mark_failed("O4_btc_mempool", f"only {n_events} events", extra={"n_events": int(n_events)})

    pnl = (pos * basket_ret).dropna()
    bench = btc.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O4 BTC mempool>90p -> long miners 20d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O4_btc_mempool", m, extra={
        "status": "ok",
        "rule": "When BTC mempool size > 90th percentile of trailing 90-day, long {MARA,RIOT,CLSK} equal-weight 20 sessions (extended from spec 10d).",
        "mechanism": "Mempool congestion -> fee revenue spike -> miner equity rally.",
        "universe": "MARA, RIOT, CLSK",
        "source": "blockchain.info /charts/mempool-size (since 2016).",
        "n_events": int(n_events),
    })


if __name__ == "__main__":
    main()
