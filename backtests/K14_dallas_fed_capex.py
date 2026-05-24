"""
K14 Dallas Fed Manufacturing Survey — Future Capex (6-months-ahead) subindex.

Data: Dallas Fed TMOS alldata_sa.xls — column 'fcexp' (future capital
expenditures, seasonally adjusted diffusion balance), monthly since 2004-06.

Rule: When MoM change in fcexp falls by > 15 points, short XLI (industrials)
for 30 trading days from the survey's end-of-month publication date.

Mechanism: A sudden drop in 6-month capex plans is an early signal of
declining industrial activity / cap-ex cycle and pressures industrials.

Source: dallasfed.org TMOS Texas Manufacturing Outlook Survey full history.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


URL = "https://www.dallasfed.org/~/media/Documents/research/surveys/tmos/documents/alldata_sa.xls"


def main():
    cache = DATA / "dallasfed_tmos_alldata_sa.xls"
    try:
        if not cache.exists():
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            cache.write_bytes(r.content)
        df = pd.read_excel(cache, sheet_name="All Data Seasonally Adjusted", header=0)
    except Exception as e:
        return mark_failed("K14_dallas_fed_capex", f"fetch/parse: {e}")

    if "fcexp" not in df.columns:
        return mark_failed("K14_dallas_fed_capex", "fcexp column missing.")
    df["Date"] = pd.to_datetime(df["Date"], format="%b-%y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    fc = pd.to_numeric(df["fcexp"], errors="coerce").dropna()
    mom = fc.diff()

    try:
        px = load_prices(["XLI", "SPY"], start="2004-06-01")
    except Exception as e:
        return mark_failed("K14_dallas_fed_capex", f"price fetch: {e}")

    rets = daily_returns(px)
    xli = rets["XLI"].dropna()

    sig = pd.Series(0.0, index=xli.index)
    n_trig = 0
    # Survey published ~last business day of the named month.
    # Use month-end of the named period as the publication date.
    for ts, dv in mom.dropna().items():
        if dv < -15:
            pub = ts + pd.offsets.MonthEnd(0)
            i = xli.index.searchsorted(pub + pd.Timedelta(days=1))
            if i >= len(xli):
                continue
            sig.iloc[i:i+30] += 1
            n_trig += 1

    pos = -(sig > 0).astype(float)  # short
    pnl = pos.shift(1) * xli
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K14 Dallas Fed capex drop short XLI")
    print_metrics(m)
    print(f"obs={len(fc)}, n_trig={n_trig}, exposure_short={(pos<0).mean():.2%}")

    save_result("K14_dallas_fed_capex", m, extra={
        "status": "ok",
        "rule": "MoM change in Dallas Fed TMOS fcexp (future capex, 6m ahead) < -15 → short XLI 30 trading days from month-end.",
        "mechanism": "Sharp drop in 6m capex plans presages slowing industrial cycle.",
        "source": "Dallas Fed TMOS alldata_sa.xls, column 'fcexp' (seasonally adjusted)",
        "n_observations": int(len(fc)),
        "n_triggers": int(n_trig),
        "exposure_pct": float((pos != 0).mean()),
    })


if __name__ == "__main__":
    main()
