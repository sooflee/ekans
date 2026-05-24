"""
G10 China port throughput vs equities (IMF PortWatch).

Data: IMF PortWatch ArcGIS FeatureServer (Daily_Ports_Data) provides daily
port-call counts per port back to ~Jan 2019. Endpoint:
  https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0/query

Method:
 1. Pull daily container port-calls for Shanghai / Ningbo / Shenzhen.
 2. Aggregate to a monthly average; compute 24-month rolling z-score.
 3. z > +1.5  => long FXI (China large-cap) vs short SPY (50/50 dollar-neutral).
    z < -1.5  => reverse (short FXI, long SPY).
    Otherwise flat.
 4. Hold until next monthly observation. Position applied to next month's
    daily returns.

Honest notes:
 - Daily endpoint paginated at 1000 rows per request — we walk through pages.
 - Coverage starts ~2019; z-score window is short (24 months minimum).
 - FXI is a noisy China proxy; not adjusting for HKD / pre-IPO drag.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

PORTWATCH_URL = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/"
                 "services/Daily_Ports_Data/FeatureServer/0/query")
PORTS = ["Shanghai", "Ningbo", "Shenzhen"]


def fetch_portwatch(ports):
    """Page through PortWatch records for the given port names."""
    rows = []
    where = "portname IN (" + ",".join(f"'{p}'" for p in ports) + ")"
    offset = 0
    batch = 1000
    while True:
        params = {
            "where": where,
            "outFields": "date,portname,portcalls,portcalls_container",
            "orderByFields": "date ASC",
            "resultRecordCount": batch,
            "resultOffset": offset,
            "f": "json",
        }
        try:
            r = requests.get(PORTWATCH_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"PortWatch query failed at offset {offset}: {e}")
        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f["attributes"]
            rows.append(a)
        if len(feats) < batch and not data.get("exceededTransferLimit"):
            break
        offset += batch
        time.sleep(0.1)
    return pd.DataFrame(rows)


def main():
    cache_fp = DATA / "portwatch_china_daily.parquet"
    df = None
    if cache_fp.exists():
        df = pd.read_parquet(cache_fp)
        # If cache is older than 7 days, refresh
        latest = pd.to_datetime(df["date"]).max()
        if (pd.Timestamp.utcnow().tz_localize(None) - latest).days > 30:
            df = None

    if df is None:
        try:
            df = fetch_portwatch(PORTS)
            if df.empty:
                return mark_failed("G10_china_port_throughput",
                                   "PortWatch returned no rows")
            df.to_parquet(cache_fp)
        except Exception as e:
            return mark_failed("G10_china_port_throughput", str(e))

    # date may be ms-since-epoch or ISO string
    if pd.api.types.is_integer_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="ms")
    else:
        df["date"] = pd.to_datetime(df["date"])

    daily = df.pivot_table(index="date", columns="portname",
                           values="portcalls_container",
                           aggfunc="sum").sort_index()
    daily = daily.fillna(0)
    # Aggregate Chinese super-cluster
    total = daily.sum(axis=1)

    # Monthly average port-calls
    monthly = total.resample("ME").mean()
    if len(monthly) < 30:
        return mark_failed("G10_china_port_throughput",
                           f"Too few monthly observations: {len(monthly)}")

    # 24-month rolling z-score (start signal once we have enough history)
    z = (monthly - monthly.rolling(24).mean()) / monthly.rolling(24).std()

    # Equities: FXI vs SPY (long/short pair)
    px = load_prices(["FXI", "SPY"], start="2018-01-01")
    rets = px.pct_change().dropna()
    spy_r = rets["SPY"]
    fxi_r = rets["FXI"]

    # Monthly position: derived at the end of month T, applied to all days in month T+1
    # +1: long FXI / short SPY
    # -1: short FXI / long SPY
    monthly_pos = pd.Series(0.0, index=z.index)
    monthly_pos[z >  1.5] =  1.0
    monthly_pos[z < -1.5] = -1.0

    # Map monthly position to daily index by forward-fill, then shift by one trading day
    daily_pos = monthly_pos.reindex(rets.index, method="ffill").fillna(0.0)

    # Pair return = pos * (fxi - spy)
    pair = (fxi_r - spy_r)
    pnl = (daily_pos.shift(1) * pair).dropna()

    if pnl.empty or pnl.abs().sum() == 0:
        return mark_failed("G10_china_port_throughput",
                           "No active signal periods in window")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index),
                        name="G10 China port throughput (FXI vs SPY)")
    print_metrics(m)
    print(f"\nMonthly signal distribution: long={int((monthly_pos>0).sum())}, "
          f"short={int((monthly_pos<0).sum())}, flat={int((monthly_pos==0).sum())}")
    print(f"Latest z-score: {z.dropna().iloc[-1]:+.2f} on {z.dropna().index[-1].date()}")

    save_result("G10_china_port_throughput", m, extra={
        "status": "ok",
        "rule": ("Monthly z-score of container-vessel port-calls at "
                 "Shanghai/Ningbo/Shenzhen (24-month window). "
                 "z>+1.5 long FXI vs short SPY; z<-1.5 reverse; else flat."),
        "data_source": "IMF PortWatch Daily_Ports_Data FeatureServer",
        "z_window_months": 24,
        "n_long_months": int((monthly_pos > 0).sum()),
        "n_short_months": int((monthly_pos < 0).sum()),
        "n_flat_months": int((monthly_pos == 0).sum()),
        "caveats": ("Data starts ~2019; short z-score window; FXI is an "
                    "imperfect China proxy."),
    })


if __name__ == "__main__":
    main()
