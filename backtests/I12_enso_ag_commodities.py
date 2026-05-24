"""
I-12 ENSO ag commodity trade (ONI -> soybeans vs corn ETFs).

Data: NOAA CPC ONI (Oceanic Niño Index) ASCII at
   https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
Monthly 3-month-running-mean Niño 3.4 anomaly since 1950.

Rule (from prompt):
  ONI > +1.0 (El Niño)  => short ZS (soybeans), long ZC (corn).
  ONI < -1.0 (La Niña)  => reverse.
  Position held while |ONI| > 0.5 (mean-reverts toward zero).
  Daily PnL on SOYB / CORN ETF proxies (yfinance).

Honest: SOYB/CORN ETFs are thin, and futures roll cost dominates. We treat
this as a low-conviction macro overlay test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Map "SEAS" code to its middle calendar month
SEAS_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def load_oni():
    cache_fp = DATA / "noaa_oni.parquet"
    try:
        r = requests.get(ONI_URL, timeout=30)
        r.raise_for_status()
        text = r.text
    except Exception:
        if cache_fp.exists():
            return pd.read_parquet(cache_fp)["oni"]
        raise
    rows = []
    for ln in text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) != 4:
            continue
        seas, yr, total, anom = parts
        mo = SEAS_TO_MONTH.get(seas.upper())
        if mo is None:
            continue
        try:
            yr_i = int(yr)
            anom_f = float(anom)
        except Exception:
            continue
        # DJF nominally spans Dec(yr-1)-Jan(yr)-Feb(yr); use mid month Jan(yr).
        # For NDJ we use Dec(yr).
        dt_ = pd.Timestamp(year=yr_i if seas.upper() != "NDJ" else yr_i,
                            month=mo, day=1)
        rows.append((dt_, anom_f))
    s = pd.DataFrame(rows, columns=["date", "oni"]).set_index("date")["oni"]
    s = s.sort_index()
    s.to_frame().to_parquet(cache_fp)
    return s


def main():
    try:
        oni = load_oni()
    except Exception as e:
        return mark_failed("I12_enso_ag_commodities", f"ONI load failed: {e}")

    if len(oni) < 60:
        return mark_failed("I12_enso_ag_commodities", "ONI series too short")

    # Build monthly state machine: state +1 (El Nino: short SOYB, long CORN),
    # state -1 (La Nina: long SOYB, short CORN), state 0 flat.
    state = pd.Series(0.0, index=oni.index)
    cur = 0
    for d in oni.index:
        v = oni.loc[d]
        if cur == 0:
            if v > 1.0:
                cur = 1
            elif v < -1.0:
                cur = -1
        elif cur == 1:
            if v < 0.5:
                cur = 0
        elif cur == -1:
            if v > -0.5:
                cur = 0
        state.loc[d] = cur

    # ETF prices: SOYB & CORN
    px = load_prices(["SOYB", "CORN"], start="2011-01-01")
    if px.empty or "SOYB" not in px.columns or "CORN" not in px.columns:
        return mark_failed("I12_enso_ag_commodities", "ETF price load failed")

    rets = px.pct_change()
    state_daily = state.reindex(rets.index, method="ffill").fillna(0)

    # state = +1 (El Nino): short SOYB, long CORN => pos = (-1 SOYB, +1 CORN)
    # state = -1 (La Nina): long SOYB, short CORN => pos = (+1 SOYB, -1 CORN)
    pos = pd.DataFrame({
        "SOYB": -0.5 * state_daily,
        "CORN":  0.5 * state_daily,
    })
    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()

    if pnl.abs().sum() == 0:
        return mark_failed("I12_enso_ag_commodities", "Signal never fired")

    bench_full = rets.mean(axis=1)
    m = compute_metrics(pnl, benchmark=bench_full.reindex(pnl.index),
                        name="I-12 ENSO (SOYB vs CORN pair)")
    print_metrics(m)
    print(f"\nState distribution (months): "
          f"el_nino={(state==1).sum()} la_nina={(state==-1).sum()} flat={(state==0).sum()}")

    save_result("I12_enso_ag_commodities", m, extra={
        "status": "ok",
        "rule": ("Monthly ONI; state machine: enter El Nino long-CORN/short-SOYB "
                 "when ONI>+1.0, exit when ONI<+0.5; reverse for La Nina. "
                 "Daily PnL on 50/50 ETF pair."),
        "data_source": "NOAA CPC ONI (3-month Nino3.4 anomaly)",
        "n_months_long_corn": int((state == 1).sum()),
        "n_months_long_soyb": int((state == -1).sum()),
        "n_months_flat": int((state == 0).sum()),
        "caveats": ("Ag ETFs (SOYB/CORN) start 2011; futures roll cost may "
                    "dominate. Low conviction."),
    })


if __name__ == "__main__":
    main()
