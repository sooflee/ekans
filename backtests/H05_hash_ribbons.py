"""
H05 Hash Ribbons (Charles Edwards).

Hash rate -> 30d SMA and 60d SMA.
  Miner capitulation event = 30d SMA crosses BELOW 60d SMA.
  Buy signal = after a capitulation, 30d SMA crosses BACK ABOVE 60d SMA.
Hold until next capitulation OR exit at +50% gain.

Data: blockchain.info charts API, 'hash-rate', timespan=all (4-day sampling).
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


def fetch_hash_rate():
    cache = DATA / "blockchain_info_hash_rate.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get(
        "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json",
        timeout=60,
    )
    j = r.json()
    rows = [(pd.Timestamp(v["x"], unit="s"), float(v["y"])) for v in j["values"]]
    df = pd.DataFrame(rows, columns=["date", "hash_rate"]).set_index("date")
    df.to_parquet(cache)
    return df


def main():
    try:
        hr = fetch_hash_rate()["hash_rate"]
    except Exception as e:
        return mark_failed("H05_hash_ribbons", f"hash-rate fetch failed: {e}")

    # Upsample to daily with ffill, then take rolling SMAs on the daily series
    hr_d = hr.resample("D").interpolate("linear")
    sma30 = hr_d.rolling(30).mean()
    sma60 = hr_d.rolling(60).mean()

    above = (sma30 > sma60).astype(int)
    # 1 -> "30 > 60" (healthy), 0 -> "30 <= 60" (capitulation)
    cross_up   = (above.diff() == 1)
    cross_down = (above.diff() == -1)

    # state machine: need a capitulation (down cross) followed by a recovery (up cross) to enter long
    # Hold long until next down cross or +50%
    px = load_prices(["BTC-USD"], start="2014-01-01").iloc[:, 0]
    idx = px.index.intersection(hr_d.index)
    px = px.reindex(idx).ffill()
    sma30 = sma30.reindex(idx).ffill()
    sma60 = sma60.reindex(idx).ffill()
    cross_up = cross_up.reindex(idx).fillna(False)
    cross_down = cross_down.reindex(idx).fillna(False)

    pos = pd.Series(0.0, index=idx)
    in_position = False
    entry_price = None
    saw_capitulation = False
    for d in idx:
        if cross_down.loc[d]:
            saw_capitulation = True
            # exit if in
            if in_position:
                in_position = False
                entry_price = None
        elif cross_up.loc[d] and saw_capitulation and not in_position:
            in_position = True
            entry_price = float(px.loc[d])
            saw_capitulation = False
        elif in_position and entry_price is not None:
            if float(px.loc[d]) >= entry_price * 1.5:
                in_position = False
                entry_price = None
        pos.loc[d] = 1.0 if in_position else 0.0

    rets = px.pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[sma60.dropna().index[0]:]

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index], name="H05 Hash Ribbons")
    print_metrics(m)
    save_result("H05_hash_ribbons", m, extra={
        "status": "ok",
        "rule": "After 30d SMA(hashrate) crosses BELOW 60d (capitulation), buy BTC on the cross BACK ABOVE 60d. Exit on next capitulation or at +50% gain.",
        "data_source": "blockchain.info charts/hash-rate (timespan=all, ~4-day sampling, interpolated to daily).",
        "n_long_days": int((pos == 1).sum()),
    })


if __name__ == "__main__":
    main()
