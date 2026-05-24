"""
H01 Perpetual funding rate extremes.

Source: Deribit BTC-PERPETUAL funding rate history (free, public).
Coverage: ~2019-05 onward, 8-hourly. (Binance API is geo-blocked from this host.)

Rule:
  - Aggregate 8h funding to daily mean.
  - Rolling 90d z-score of daily funding.
  - z > +2 for 5+ consecutive days  -> short BTC for 7 calendar days.
  - z < -2 for 5+ consecutive days  -> long  BTC for 7 calendar days.
  - Overlapping signals: latest wins (state machine).
PnL on BTC-USD daily returns.
"""
import sys, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import numpy as np
import pandas as pd

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, rolling_zscore, DATA,
)


CACHE = DATA / "deribit_btc_funding.parquet"


def fetch_deribit_funding():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    # Walk forward 30-day windows from 2019-05-01 to now
    start = int(pd.Timestamp("2019-05-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp.utcnow().timestamp() * 1000)
    rows = []
    step = 25 * 86400 * 1000  # 25 days per call to stay under 744 row API limit
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        try:
            r = requests.get(
                "https://www.deribit.com/api/v2/public/get_funding_rate_history",
                params={"instrument_name": "BTC-PERPETUAL",
                        "start_timestamp": cur, "end_timestamp": nxt},
                timeout=30,
            )
            j = r.json()
            for x in j.get("result", []):
                rows.append((x["timestamp"], x.get("interest_8h"), x.get("index_price")))
        except Exception as e:
            print("err", e, "at", cur)
        cur = nxt
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["ts", "interest_8h", "index_price"]).drop_duplicates("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.sort_values("dt").set_index("dt")
    df.to_parquet(CACHE)
    return df


def main():
    try:
        fund = fetch_deribit_funding()
    except Exception as e:
        return mark_failed("H01_perp_funding", f"funding fetch failed: {e}")

    if fund.empty or fund["interest_8h"].isna().all():
        return mark_failed("H01_perp_funding", "no funding data returned")

    # daily mean funding (8h funding rate is per 8h; sum over a day ~= daily funding)
    daily_funding = fund["interest_8h"].astype(float).resample("D").mean().dropna()

    z = rolling_zscore(daily_funding, 90)

    # consecutive-extreme counters
    hi = (z > 2).astype(int)
    lo = (z < -2).astype(int)
    hi_run = hi.groupby((hi == 0).cumsum()).cumcount() + 1
    hi_run = hi_run.where(hi == 1, 0)
    lo_run = lo.groupby((lo == 0).cumsum()).cumcount() + 1
    lo_run = lo_run.where(lo == 1, 0)

    sig_short = (hi_run >= 5)
    sig_long  = (lo_run >= 5)

    pos = pd.Series(0.0, index=daily_funding.index)
    hold_days = 7
    cur_pos = 0.0
    cur_left = 0
    for d in pos.index:
        if sig_short.get(d, False):
            cur_pos = -1.0
            cur_left = hold_days
        elif sig_long.get(d, False):
            cur_pos = 1.0
            cur_left = hold_days
        pos.loc[d] = cur_pos
        cur_left -= 1
        if cur_left <= 0:
            cur_pos = 0.0
            cur_left = 0

    px = load_prices(["BTC-USD"], start="2014-01-01")
    rets = px.iloc[:, 0].pct_change().dropna()

    pos_d = pos.reindex(rets.index, method="ffill").fillna(0.0)
    pnl = (pos_d.shift(1) * rets).dropna()

    # Trim to period with actual signals
    pnl = pnl.loc[daily_funding.index[0]:]

    m = compute_metrics(pnl, benchmark=rets.loc[pnl.index], name="H01 Perp funding extremes")
    print_metrics(m)
    save_result("H01_perp_funding", m, extra={
        "status": "ok",
        "rule": "z90(8h funding mean) > +2 for 5+d -> short BTC 7d; z < -2 for 5+d -> long BTC 7d.",
        "data_source": "Deribit BTC-PERPETUAL funding (interest_8h) since 2019-05",
        "note": "Binance funding API is geo-blocked from this host; Deribit substituted.",
        "n_signals_short": int((sig_short).sum()),
        "n_signals_long": int((sig_long).sum()),
    })


if __name__ == "__main__":
    main()
