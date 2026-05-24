"""
T5 Henry Hub - Waha basis -> NG=F.

Rule: When (Waha - HH) < -$3 / MMBtu for 5 consecutive days, long NG=F for 4 weeks.

Data path tried:
  - HH daily spot: FRED DHHNGSP works (1997->today).
  - HH daily spot also EIA hist_xls/RNGWHHDd.xls (works without key).
  - Waha daily spot:
      * Not on FRED (no DWAHANGSP).
      * EIA legacy hist_xls/RNGWA0d.xls returns 404.
      * EIA v2 API requires api_key (RNGWA0.D).
  - Without an EIA_API_KEY env var, Waha daily history is not freely scrapable.

If EIA_API_KEY is set, we pull Waha via EIA v2 API. Otherwise we mark_failed
and leave a stub.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


def fetch_waha_eia(api_key, start="2010-01-01"):
    """Pull Waha daily natural gas spot from EIA v2 API.
    series id: NG.RNGWA0.D
    """
    url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
    out = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[duoarea][]": "RGC",  # Permian/Waha area code
            "facets[product][]": "EPG0",
            "start": start,
            "offset": offset,
            "length": 5000,
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        rows = j.get("response", {}).get("data", [])
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 5000:
            break
        offset += 5000
    if not out:
        return None
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["period"])
    s = pd.Series(df["value"].astype(float).values, index=df["date"], name="WAHA")
    return s.sort_index()


def main():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        return mark_failed(
            "T5_hh_waha_basis",
            "Waha daily spot price requires EIA v2 API (api_key). "
            "Not on FRED (no DWAHANGSP); EIA hist_xls archive returns 404 for RNGWA0d.xls. "
            "Set EIA_API_KEY env var to enable.",
            extra={
                "rule": "(Waha - HH) < -$3/MMBtu for 5 days -> long NG=F 4 weeks",
                "mechanism": "Permian gas takeaway constraint resolution -> HH normalization upside",
                "source": "EIA NG.RNGWA0.D (auth-walled) + FRED DHHNGSP",
            }
        )

    # If key is present, pull and run.
    try:
        hh = load_fred("DHHNGSP", start="2010-01-01").iloc[:, 0].rename("HH")
    except Exception as e:
        return mark_failed("T5_hh_waha_basis", f"FRED DHHNGSP failed: {e}")
    waha = fetch_waha_eia(api_key)
    if waha is None or len(waha) < 250:
        return mark_failed("T5_hh_waha_basis", "EIA Waha fetch returned empty / too short")

    basis = (waha - hh).dropna()
    cond = basis < -3.0
    sig = cond & cond.shift(1) & cond.shift(2) & cond.shift(3) & cond.shift(4)

    px = load_prices(["NG=F"], start="2010-01-01")
    ng = px["NG=F"].dropna()
    ret = ng.pct_change()

    HOLD = 20
    pos = pd.Series(0.0, index=ng.index)
    for d in sig[sig].index:
        i = ng.index.searchsorted(d) + 1
        if i >= len(ng.index):
            continue
        end = min(i + HOLD, len(ng.index))
        pos.iloc[i:end] = 1.0

    pnl = (pos.shift(1) * ret).dropna()
    bench = ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T5 HH-Waha basis -> long NG=F")
    print_metrics(m)
    save_result("T5_hh_waha_basis", m, extra={
        "status": "ok",
        "rule": "(Waha - HH) < -$3/MMBtu for 5 days -> long NG=F 4 weeks",
        "mechanism": "Permian takeaway congestion resolves -> upstream curtails -> HH bid",
        "source": "EIA v2 (RNGWA0.D) + FRED DHHNGSP, NG=F yfinance",
        "n_triggers": int(sig.sum()),
    })


if __name__ == "__main__":
    main()
