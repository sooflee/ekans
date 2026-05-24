"""
T3 Indian monsoon -> sugar.

Original idea: IMD daily rainfall; when monsoon is weak, long SB=F Jul->Oct.

Substitution: IMD daily historical district-level rainfall is on
mausam.imd.gov.in but requires a multi-step paid request for archive.
We use NOAA ONI (Oceanic Nino Index) seasonal anomalies as a proxy:
positive ONI (El Nino) is empirically associated with weak SW monsoon in India.

Data: NOAA CPC ONI ascii table (oni.ascii.txt).
Rule:
  For each year, look at JJA ONI (Jun-Jul-Aug) anomaly.
  If JJA ONI > +0.5 (weak/El Nino-leaning monsoon), enter long SB=F
  on July 1 (or first SB trading day at/after July 1) of that year and
  hold through October 31.
"""
import sys
import io
import requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def fetch_oni():
    cache_fp = DATA / "cpc_oni.parquet"
    if cache_fp.exists():
        return pd.read_parquet(cache_fp)
    r = requests.get(ONI_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    rows = []
    for ln in r.text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) != 4:
            continue
        seas, yr, _total, anom = parts
        try:
            rows.append({"seas": seas, "yr": int(yr), "anom": float(anom)})
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    df.to_parquet(cache_fp)
    return df


def main():
    oni = fetch_oni()
    if oni is None or oni.empty:
        return mark_failed("T3_indian_monsoon_sugar", "ONI fetch failed")

    # JJA = Jun-Jul-Aug. Use JAS (Jul-Aug-Sep) as the indicator that's
    # actually known by the time we enter July (NOAA publishes ~mid-month
    # following). For tradability, use prior season JJF that is known by July:
    # actually the published JJA value is the centered Jul-Aug-Sep average.
    # We'll use MJJ (May-Jun-Jul) - this is announced around mid-July and
    # captures whether ENSO is El Nino-leaning at monsoon onset.
    target = oni[oni["seas"] == "MJJ"].copy()
    el_nino_years = target[target["anom"] > 0.5]["yr"].astype(int).tolist()
    print("MJJ El Nino years (ONI > +0.5):", el_nino_years)

    try:
        sb = load_prices(["SB=F"], start="2000-01-01").iloc[:, 0].rename("SB")
    except Exception as e:
        return mark_failed("T3_indian_monsoon_sugar", f"price load: {e}")
    if sb.empty:
        return mark_failed("T3_indian_monsoon_sugar", "SB=F empty")
    sb_r = sb.pct_change()

    pos = pd.Series(0.0, index=sb.index)
    trades = []
    for y in el_nino_years:
        start = pd.Timestamp(f"{y}-07-15")  # mid-July (after MJJ is announced)
        end = pd.Timestamp(f"{y}-10-31")
        mask = (sb.index >= start) & (sb.index <= end)
        if mask.any():
            pos[mask] = 1.0
            trades.append({"year": int(y), "from": str(start.date()), "to": str(end.date())})

    pnl = (pos.shift(1) * sb_r).dropna()
    bench = sb_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T3 Weak monsoon (MJJ ONI>+0.5) -> long SB=F Jul-Oct")
    print_metrics(m)
    print(f"trade-years: {len(trades)}")

    save_result("T3_indian_monsoon_sugar", m, extra={
        "status": "ok",
        "rule": ("If NOAA CPC MJJ ONI > +0.5 (El Nino-leaning at monsoon "
                 "onset), long SB=F July 15 -> Oct 31."),
        "mechanism": ("El Nino correlates with weak Indian SW monsoon -> "
                      "lower cane yield in Maharashtra/UP -> India shifts "
                      "from net exporter to net importer -> global sugar bid."),
        "source": "NOAA CPC ONI ascii table; yfinance SB=F",
        "trades": trades,
        "n_trades": len(trades),
        "caveats": ("ENSO is a proxy for monsoon, not actual rainfall. "
                    "IMD historical daily rainfall is paywalled. "
                    "El Nino does not deterministically yield weak monsoon."),
    })


if __name__ == "__main__":
    main()
