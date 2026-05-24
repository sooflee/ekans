"""
K10 China Rare Earth Exports — long REMX on supply tightening.

Data: UN Comtrade public preview API.
Reporter = 156 (China), commodity HS 2846 (rare-earth metals & compounds),
flow X (exports), partner 0 (world total). Monthly observations.

Rule: When monthly China rare-earth exports (kg, netWgt) drops > 25% YoY,
go long REMX (VanEck Rare Earth & Strategic Metals ETF) for 60 trading
days starting end of the month following the data month (Chinese customs
usually publishes monthly stats with ~3-4 week lag).

Mechanism: A material YoY decline in Chinese REE exports tightens global
supply, lifting non-China REE producer prices and REMX.

Source: comtradeapi.un.org public preview endpoint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


def fetch_china_ree():
    cache = DATA / "comtrade_china_ree.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    # Comtrade preview accepts up to ~12 periods per call.
    for yr in range(2010, 2026):
        periods = ",".join(f"{yr}{m:02d}" for m in range(1, 13))
        url = ("https://comtradeapi.un.org/public/v1/preview/C/M/HS"
               f"?reporterCode=156&period={periods}&cmdCode=2846&flowCode=X&partnerCode=0")
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"warn {yr}: status {r.status_code}")
                continue
            data = r.json().get("data", [])
            rows.extend(data)
        except Exception as e:
            print(f"warn {yr}: {e}")
        time.sleep(1.5)  # gentle on the rate limit
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_parquet(cache)
    return df


def main():
    try:
        df = fetch_china_ree()
    except Exception as e:
        return mark_failed("K10_china_rare_earth", f"Comtrade fetch: {e}")
    if df.empty:
        return mark_failed("K10_china_rare_earth", "Comtrade returned empty.")

    df["period"] = df["period"].astype(str)
    df["date"] = pd.to_datetime(df["period"] + "01", format="%Y%m%d")
    df = df.dropna(subset=["netWgt"]).sort_values("date").set_index("date")
    monthly = df["netWgt"].astype(float).resample("MS").sum()
    yoy = monthly.pct_change(12)

    try:
        px = load_prices(["REMX", "SPY"], start="2010-10-27")  # REMX inception 2010-10
    except Exception as e:
        return mark_failed("K10_china_rare_earth", f"price fetch: {e}")
    rets = daily_returns(px)
    remx = rets["REMX"].dropna()

    sig = pd.Series(0.0, index=remx.index)
    n_trig = 0
    for ts, v in yoy.dropna().items():
        if v < -0.25:
            # publication ~25 days after month-end → start trading then
            pub = ts + pd.offsets.MonthEnd(0) + pd.Timedelta(days=30)
            i = remx.index.searchsorted(pub)
            if i >= len(remx):
                continue
            sig.iloc[i:i+60] += 1
            n_trig += 1
    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * remx
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K10 China REE drop long REMX")
    print_metrics(m)
    print(f"months={len(monthly)}, n_trig={n_trig}, exposure={pos.mean():.2%}")

    save_result("K10_china_rare_earth", m, extra={
        "status": "ok",
        "rule": "Monthly China REE exports (HS 2846, kg) YoY change < -25% → long REMX 60 trading days starting ~1 month after data-month-end.",
        "mechanism": "China supplies a majority of global REE; export contractions lift global prices and benefit non-China producers.",
        "source": "UN Comtrade public preview API (reporterCode=156, cmdCode=2846, flow=X, partnerCode=0)",
        "months_observed": int(len(monthly)),
        "n_triggers": int(n_trig),
        "exposure_pct": float(pos.mean()),
    })


if __name__ == "__main__":
    main()
