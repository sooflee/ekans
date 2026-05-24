"""
H09 Spot BTC ETF net flows.

Target source (farside.co.uk/btc/) is behind a Cloudflare bot challenge from
this host and returns 403 even with browser-like UAs. Other free aggregators
(sosovalue, bitmex research) are similarly blocked or require keys.

We approximate spot BTC ETF DAILY NET FLOWS using a free yfinance proxy:
  flow_t ≈ (close_t * volume_t) summed across the 10 US spot BTC ETFs
  IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, GBTC.

This is *dollar volume*, not the actual creation/redemption net flow. It is a
noisy but broadly directional proxy: high gross dollar volume on up-days
correlates with net creations, and vice versa. We sign it by the BTC return:

  signed_flow_t = sign(BTC_return_t) * dollar_volume_t

then 5-day sum.

Rule:
  - 5-day signed-flow sum > +$5B  -> long BTC
  - 3 consecutive days of signed-flow sum < -$2.5B -> flat
Only after Jan 11 2024 (first US spot BTC ETF trading day).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

ETFS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "GBTC"]


def main():
    import yfinance as yf
    cache = DATA / "btc_spot_etfs_dollar_volume.parquet"
    if not cache.exists():
        df = yf.download(ETFS, start="2024-01-01", progress=False, auto_adjust=True)
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
        vol = df["Volume"] if isinstance(df.columns, pd.MultiIndex) else df[["Volume"]]
        dv = (close * vol).sum(axis=1).to_frame("dollar_volume")
        dv.to_parquet(cache)
    dv = pd.read_parquet(cache)["dollar_volume"]

    btc = load_prices(["BTC-USD"], start="2024-01-01").iloc[:, 0]
    btc_d = btc.reindex(dv.index, method="ffill")
    btc_ret = btc_d.pct_change()

    signed = np.sign(btc_ret) * dv
    sum5 = signed.rolling(5).sum()
    sum1 = signed

    # 3 consecutive days where rolling-1 (i.e. that day's signed flow) < -2.5B
    neg = (sum1 < -2.5e9).astype(int)
    neg_run = neg.groupby((neg == 0).cumsum()).cumcount() + 1
    neg_run = neg_run.where(neg == 1, 0)

    state = 0
    pos = pd.Series(0.0, index=dv.index)
    for d in dv.index:
        s = sum5.get(d, np.nan)
        if pd.notna(s) and s > 5e9:
            state = 1
        if neg_run.get(d, 0) >= 3:
            state = 0
        pos.loc[d] = state

    pnl = (pos.shift(1) * btc_ret).dropna()

    m = compute_metrics(pnl, benchmark=btc_ret.loc[pnl.index],
                        name="H09 BTC spot ETF dollar-volume proxy")
    print_metrics(m)
    save_result("H09_etf_flows", m, extra={
        "status": "ok",
        "rule": "5d sum of sign(BTC ret)*dollar-volume across 10 spot BTC ETFs > $5B -> long; 3 consecutive days < -$2.5B -> flat.",
        "data_source": "yfinance dollar-volume across IBIT/FBTC/BITB/ARKB/BTCO/EZBC/BRRR/HODL/BTCW/GBTC. Farside.co.uk is Cloudflare-blocked.",
        "etfs": ETFS,
        "note": "PROXY: dollar volume signed by BTC return, NOT the true creation/redemption net flow series. Brief allowed proxies when paywalled.",
    })


if __name__ == "__main__":
    main()
