"""
V11 Sidial dispersion trade
Data:
  CBOE COR3M (3-month S&P 500 implied correlation index) — historical CSV is
  served at cdn.cboe.com.
  ^VVIX and ^VIX from yfinance.
Rule:
  When COR3M < trailing-1y 20th percentile AND VVIX/VIX > 6.5, open the
  dispersion proxy: long QQQ, short SPY, equal vol-weighted (target gross
  exposure ~ 1.0).
  Vol-weighting: weight_i ∝ 1 / sigma_i (21d realised vol). Then normalise so
  total notional = 1 long, 1 short.
  Hold while both conditions remain; flat otherwise.
Mechanism (Imran Sidial / dispersion traders): low implied correlation with
high vol-of-vol signals that index vol is cheap relative to single-name vol —
classic setup for dispersion (long single names / short index). QQQ vs SPY
captures the cleanest US-listed dispersion proxy (concentrated single-name
risk vs broad index).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")

from harness import (
    load_prices, compute_metrics, print_metrics,
    save_result, mark_failed,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR3M_History.csv"
CACHE = DATA_DIR / "cboe_cor3m.parquet"


def load_cor3m(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)["COR3M"]
    r = requests.get(CBOE_URL, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    s = df["CLOSE"].astype(float).rename("COR3M").to_frame()
    s.to_parquet(CACHE)
    return s["COR3M"]


def main():
    try:
        cor3m = load_cor3m()
    except Exception as e:
        return mark_failed("V11_sidial_dispersion", f"COR3M fetch failed: {e}")

    try:
        px = load_prices(["QQQ", "SPY", "^VIX", "^VVIX"], start="2007-03-01")
    except Exception as e:
        return mark_failed("V11_sidial_dispersion", f"yfinance load: {e}")
    px = px.dropna(how="all")

    df = pd.concat([
        cor3m,
        px["QQQ"].rename("QQQ"), px["SPY"].rename("SPY"),
        px["^VIX"].rename("VIX"), px["^VVIX"].rename("VVIX"),
    ], axis=1).dropna()
    if df.empty:
        return mark_failed("V11_sidial_dispersion", "no COR3M/QQQ/SPY/VIX/VVIX overlap")

    # rolling 1y 20th pct of COR3M
    pct20 = df["COR3M"].rolling(252).quantile(0.20)
    vvix_vix = df["VVIX"] / df["VIX"]
    trig = (df["COR3M"] < pct20) & (vvix_vix > 6.5)

    qqq_r = df["QQQ"].pct_change()
    spy_r = df["SPY"].pct_change()
    qqq_vol = qqq_r.rolling(21).std()
    spy_vol = spy_r.rolling(21).std()

    # Equal-vol weights for the dispersion pair: long QQQ at (1/qqq_vol), short SPY at (1/spy_vol),
    # normalised so each leg is unit notional in vol units (gross ~2). We then scale gross to 1
    # so that strategy comparable to other long-short pairs.
    inv_qq = 1.0 / qqq_vol
    inv_sp = 1.0 / spy_vol
    norm = (inv_qq + inv_sp)
    w_qqq = (inv_qq / norm).where(trig, 0.0)
    w_spy = -(inv_sp / norm).where(trig, 0.0)

    pos = pd.concat([w_qqq.rename("QQQ"), w_spy.rename("SPY")], axis=1)
    rets = pd.concat([qqq_r.rename("QQQ"), spy_r.rename("SPY")], axis=1)
    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()

    bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V11 Sidial dispersion proxy")
    print_metrics(m)
    save_result("V11_sidial_dispersion", m, extra={
        "status": "ok",
        "rule": "When COR3M < 1y 20th pct AND VVIX/VIX > 6.5 -> long QQQ / short SPY, equal vol-weighted (21d realised vol), gross ~1. Flat otherwise.",
        "mechanism": "Low implied correlation + high vol-of-vol means index vol is cheap vs single-name vol — classic dispersion setup; QQQ vs SPY is the cleanest US-listed proxy.",
        "source": "Imran Sidial, YouTube interview round 2 (Phase 1V).",
        "data_source": "CBOE COR3M CSV (cdn.cboe.com); yfinance for QQQ/SPY/VIX/VVIX.",
        "pct_time_on": float(trig.reindex(pnl.index).mean()),
    })


if __name__ == "__main__":
    main()
