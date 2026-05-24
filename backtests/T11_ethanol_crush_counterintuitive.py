"""
T11 Ethanol crush counterintuitive -> long corn.

Original rule: Ethanol production drops > 5% WoW AND ethanol-corn crush margin
negative for 2 consecutive weeks -> long ZC=F for 6 weeks.

Mechanism: production cuts unwind corn demand, BUT historically the offset
(DDG -> feed displacement; refiners resume crushing on margin recovery)
manifests as a rebound in corn the following weeks. We're testing the
short-horizon contrarian play.

Data:
  - Ethanol weekly production: EIA hist_xls W_EPOOXE_YOP_NUS_MBBLDw.xls (free).
  - Ethanol price: EH=F (CME ethanol futures) - yfinance.
  - Corn price: ZC=F - yfinance.
  - DDG (Distillers Dried Grains): no free public daily series; we proxy
    crush margin as eth_revenue - corn_cost without DDG credit.
       1 bu corn -> ~2.8 gal ethanol
       crush ($/bu) = EH_price * 2.8 - ZC_price/100
       (EH=F in $/gal; ZC=F in c/bu, hence /100)
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

EIA_URL = ("https://www.eia.gov/dnav/pet/hist_xls/"
           "W_EPOOXE_YOP_NUS_MBBLDw.xls")


def fetch_eia_weekly_ethanol():
    cache_fp = DATA / "eia_weekly_ethanol_oxe.parquet"
    if cache_fp.exists():
        s = pd.read_parquet(cache_fp).iloc[:, 0]
        return s
    r = requests.get(EIA_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "kbd"]
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(cache_fp)
    return df["kbd"]


def main():
    eth_prod = fetch_eia_weekly_ethanol()
    if eth_prod is None or len(eth_prod) < 100:
        return mark_failed("T11_ethanol_crush_counterintuitive",
                           "EIA weekly ethanol production fetch failed")

    try:
        eh = load_prices(["EH=F"], start="2010-06-01").iloc[:, 0].rename("EH")
        zc = load_prices(["ZC=F"], start="2010-06-01").iloc[:, 0].rename("ZC")
    except Exception as e:
        return mark_failed("T11_ethanol_crush_counterintuitive",
                           f"price load failed: {e}")

    eh = eh.dropna()
    zc = zc.dropna()
    if len(eh) < 250 or len(zc) < 250:
        return mark_failed("T11_ethanol_crush_counterintuitive",
                           "insufficient EH=F / ZC=F history")

    # Daily crush ($/bu): EH ($/gal) * 2.8 - ZC (c/bu)/100
    px = pd.concat([eh, zc], axis=1).ffill().dropna()
    crush = px["EH"] * 2.8 - px["ZC"] / 100.0
    crush.name = "crush"

    # Weekly crush ending each Friday (last value)
    crush_w = crush.resample("W-FRI").last()
    # Weekly production WoW
    prod_w = eth_prod.reindex(crush_w.index, method="ffill")
    prod_wow = prod_w.pct_change()

    cond_prod = prod_wow < -0.05
    cond_crush = (crush_w < 0) & (crush_w.shift(1) < 0)

    sig_w = cond_prod & cond_crush
    n_sig = int(sig_w.sum())

    HOLD = 30  # 6 weeks
    zc_r = zc.pct_change()
    pos = pd.Series(0.0, index=zc.index)
    for d in sig_w[sig_w].index:
        i = zc.index.searchsorted(d) + 1
        if i >= len(zc.index):
            continue
        end = min(i + HOLD, len(zc.index))
        pos.iloc[i:end] = 1.0

    pnl = (pos.shift(1) * zc_r).dropna()
    bench = zc_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T11 Ethanol crush counterintuitive -> long ZC=F")
    print_metrics(m)
    print(f"triggers: {n_sig}")

    save_result("T11_ethanol_crush_counterintuitive", m, extra={
        "status": "ok",
        "rule": ("Weekly EIA ethanol production WoW < -5% AND crush margin "
                 "(EH*2.8 - ZC/100) negative for 2 weeks -> long ZC=F 6 weeks."),
        "mechanism": ("Production cut -> short-term corn demand drop -> price "
                      "weakness draws buyers; DDG credit channel ignored."),
        "source": "EIA weekly oxygenate production (xls), yfinance EH=F, ZC=F",
        "n_triggers": n_sig,
        "caveats": ("DDG (distillers grains) revenue not modelled; crush is "
                    "understated. EH=F is thinly traded with sparse settlement "
                    "history pre-2015."),
    })


if __name__ == "__main__":
    main()
