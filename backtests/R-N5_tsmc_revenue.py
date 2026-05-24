"""
R-N5 TSMC revenue (retry).

Original spec: long NVDA+AVGO+AMD+ASML basket 10d when TSMC monthly YoY
beats trailing-3m YoY avg by 8pp.

TSMC investor.tsmc.com is 403 from this sandbox. yfinance TSM only exposes
~7 quarters. Substitution: scrape Macrotrends quarterly revenue (60q
history, ~15y) and run the analogous test on quarterly cadence:
  Signal: when TSMC quarterly YoY > trailing-3-quarter YoY mean + 8pp,
  go long the basket for the following ~63 sessions (~1 quarter).
"""
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save_result, mark_failed, compute_metrics, load_prices, DATA


SIGNAL_ID = "R-N5_tsmc_revenue"
BASKET = ["NVDA", "AVGO", "AMD", "ASML"]


def fetch_tsmc_quarterly():
    fp = DATA / "tsmc_quarterly_revenue.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    r = requests.get(
        "https://www.macrotrends.net/stocks/charts/TSM/taiwan-semiconductor-manufacturing/revenue",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    tables = pd.read_html(io.StringIO(r.text))
    # table[1] is quarterly
    df = tables[1].copy()
    df.columns = ["date", "revenue"]
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["revenue"].str.replace(r"[\$,]", "", regex=True).astype(float)
    df = df.sort_values("date").set_index("date")
    df.to_parquet(fp)
    return df


def main():
    try:
        rev = fetch_tsmc_quarterly()
    except Exception as e:
        return mark_failed(SIGNAL_ID, f"Macrotrends scrape failed: {e}")
    print(f"  TSMC quarterly: {len(rev)} rows, {rev.index[0].date()} to {rev.index[-1].date()}")

    rev["yoy"] = rev["revenue"].pct_change(4) * 100
    rev["yoy_3q_mean"] = rev["yoy"].shift(1).rolling(3).mean()
    rev["beat"] = rev["yoy"] - rev["yoy_3q_mean"]
    rev = rev.dropna()
    print(f"  with YoY beat: {len(rev)} quarters")

    # Get basket prices
    start = rev.index[0].date().isoformat()
    px = load_prices(BASKET, start=start)
    px = px.dropna(how="all")
    print(f"  basket prices shape: {px.shape}")
    # Basket = equal-weighted daily return
    ret = px.pct_change()
    basket_ret = ret.mean(axis=1)

    # Signal: beat > 8pp -> long basket for next ~63 sessions (~1 quarter)
    threshold_pp = 8.0
    signal = (rev["beat"] > threshold_pp)
    n_events = int(signal.sum())
    print(f"  events (beat > {threshold_pp}pp): {n_events}")

    # When signal fires on quarter date, hold long for ~63 trading days starting at signal date
    pos = pd.Series(0.0, index=basket_ret.index)
    sig_dates = signal[signal].index
    for sd in sig_dates:
        # Find the next trading day >= sd
        loc = pos.index.searchsorted(sd)
        if loc >= len(pos):
            continue
        for k in range(0, 63):
            if loc + k < len(pos):
                # overlay (cap to 1.0 to avoid stacking)
                pos.iloc[loc + k] = 1.0

    # Apply with shift-1 to remove same-day look-ahead
    pos = pos.shift(1).fillna(0.0)
    pnl = pos * basket_ret

    metrics = compute_metrics(pnl.dropna(), benchmark=basket_ret,
                              name="TSMC YoY-beat > 8pp long semi-basket ~1q")
    print("Metrics:", metrics)

    extra = {
        "rule": "Long equal-weighted NVDA/AVGO/AMD/ASML basket ~63 sessions when TSMC quarterly YoY exceeds trailing 3q YoY mean by >= 8pp.",
        "mechanism": "TSMC is the bellwether foundry; revenue surprises lead semi customer/supplier basket performance via demand pull-through.",
        "source": "Macrotrends TSM quarterly revenue (scrape); yfinance prices.",
        "n_events": n_events,
        "data_substitution": "TSMC IR monthly revenue page returns 403; yfinance exposes only ~7 quarters. Substituted Macrotrends quarterly history (~60 quarters, ~15y). Holding period 1q replaces 10-day monthly cadence.",
        "status": "ok",
    }
    save_result(SIGNAL_ID, metrics, extra=extra)
    return metrics


if __name__ == "__main__":
    main()
