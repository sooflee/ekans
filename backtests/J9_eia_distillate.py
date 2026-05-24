"""
J9 EIA distillate stocks anomaly -> long XLE / short USO.

EIA Open Data API requires an API key. We instead pull the public weekly XLS
from EIA DNAV: https://www.eia.gov/dnav/pet/hist_xls/WDISTUS1w.xls
("Weekly U.S. Ending Stocks of Distillate Fuel Oil")

Rule:
- For each week, compute the same-week-of-year 5-year rolling mean and std (using
  prior 5 same-weeks).
- z = (cur - mean) / std. When z < -2.0, treat as draw event: long XLE / short USO,
  hold 10 trading days (2 weeks). PnL = XLE_ret - USO_ret.
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)


def fetch_eia_distillate():
    cache = DATA / "eia_wdistus1.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = "https://www.eia.gov/dnav/pet/hist_xls/WDISTUS1w.xls"
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # The XLS has metadata sheet then "Data 1" with weekly series
    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, engine="xlrd")
    # Find sheet with the time series (header in row 2)
    df = None
    for name, sh in sheets.items():
        if "Data" in name:
            try:
                sub = pd.read_excel(io.BytesIO(r.content), sheet_name=name, skiprows=2, engine="xlrd")
                if sub.shape[1] >= 2 and sub.shape[0] > 100:
                    df = sub
                    break
            except Exception:
                continue
    if df is None:
        # fallback: first sheet
        df = list(sheets.values())[0]
    df.columns = ["date", "stocks_kbbl"] + list(df.columns[2:])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")[["stocks_kbbl"]]
    df["stocks_kbbl"] = pd.to_numeric(df["stocks_kbbl"], errors="coerce")
    df = df.dropna().sort_index()
    df.to_parquet(cache)
    return df


def main():
    try:
        df = fetch_eia_distillate()
    except Exception as e:
        return mark_failed("J9_eia_distillate", f"EIA fetch failed: {e}")
    if df.empty:
        return mark_failed("J9_eia_distillate", "Empty EIA data")

    s = df["stocks_kbbl"]
    # same-week-of-year 5-year mean & std (use prior 5 same weeks)
    s_df = s.to_frame("stocks")
    s_df["woy"] = s_df.index.isocalendar().week
    s_df["year"] = s_df.index.year

    # For each row, the prior 5 same-WoY readings
    means = {}
    stds = {}
    by_woy = {w: g.sort_index() for w, g in s_df.groupby("woy")}
    for d, row in s_df.iterrows():
        w = row["woy"]
        g = by_woy[w]
        prior = g[g.index < d].tail(5)["stocks"]
        if len(prior) >= 3:
            means[d] = prior.mean()
            stds[d] = prior.std()
    seasonal_mu = pd.Series(means)
    seasonal_sd = pd.Series(stds)
    z = (s.reindex(seasonal_mu.index) - seasonal_mu) / seasonal_sd
    triggers = z[z < -2.0].index

    px = load_prices(["XLE", "USO"], start="2007-01-01")
    if px.empty or "XLE" not in px.columns or "USO" not in px.columns:
        return mark_failed("J9_eia_distillate", "XLE/USO load failed")
    rets = px.pct_change()

    daily_pos_xle = pd.Series(0.0, index=rets.index)
    daily_pos_uso = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 10, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos_xle.iloc[j] = 1.0
            daily_pos_uso.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    pnl = (daily_pos_xle.shift(1) * rets["XLE"] + daily_pos_uso.shift(1) * rets["USO"]).dropna()
    bench = rets["XLE"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J9 Distillate draw -> long XLE / short USO")
    print_metrics(m)
    print(f"\nTrigger events: {n_events}")

    save_result("J9_eia_distillate", m, extra={
        "status": "ok",
        "rule": ("EIA weekly distillate stocks z-score (vs prior 5 same-weeks) < -2.0 "
                 "-> long XLE / short USO for 10 trading days (crack-spread proxy)."),
        "mechanism": "Distillate draws below seasonal range signal refining-product tightness; refiners (XLE-weighted) widen crack spreads while crude (USO) lags.",
        "source": "https://www.eia.gov/dnav/pet/hist_xls/WDISTUS1w.xls (EIA Weekly Distillate Stocks)",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
