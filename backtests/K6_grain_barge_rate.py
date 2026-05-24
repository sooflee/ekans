"""
K6 USDA Grain Barge Rate (St. Louis) — short MOO on WoW surges.

Data: USDA AMS Grain Transportation Report, Figure 10/Table 9 workbook,
sheet 'TWK' — weekly barge spot rates expressed as percent of tariff,
column 'ST LOUIS', from 2004 onward. This is the standard published GTR
percent-of-tariff series for St. Louis (the original signal called for
St-Louis-to-NOLA barge rate as % of tariff; the GTR provides St. Louis
% of tariff as the headline series and "to NOLA" is the implied dry
bulk flow).

Rule: When weekly ST LOUIS % of tariff surges > 30% WoW, short MOO
(VanEck Agribusiness ETF) for 15 trading days starting the Tuesday
following the GTR release (GTR is released each Thursday for the
prior week ending Tuesday).

Mechanism: Barge-rate spikes signal river capacity stress (low water,
demand surge) that compresses agribusiness margins (exporters/processors
held in MOO).

Source: ams.usda.gov GTRFigure10Table9.xlsx 'TWK' sheet, ST LOUIS column.
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


URL = "https://www.ams.usda.gov/sites/default/files/media/GTRFigure10Table9.xlsx"


def main():
    cache = DATA / "gtr_fig10_table9.xlsx"
    try:
        if not cache.exists():
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            cache.write_bytes(r.content)
        df = pd.read_excel(cache, sheet_name="TWK", header=0)
    except Exception as e:
        return mark_failed("K6_grain_barge_rate", f"fetch/parse: {e}")

    df = df[["DATE", "ST LOUIS"]].dropna()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")
    rate = df["ST LOUIS"].astype(float)
    wow = rate.pct_change()

    try:
        px = load_prices(["MOO", "SPY"], start="2007-09-01")  # MOO inception Sep 2007
    except Exception as e:
        return mark_failed("K6_grain_barge_rate", f"price fetch: {e}")
    rets = daily_returns(px)
    moo = rets["MOO"].dropna()

    sig = pd.Series(0.0, index=moo.index)
    n_trig = 0
    for ts, v in wow.dropna().items():
        if v > 0.30:
            # GTR released next Thursday after Tue rate; trade next biz day
            pub = ts + pd.Timedelta(days=2)
            i = moo.index.searchsorted(pub)
            if i >= len(moo):
                continue
            sig.iloc[i:i+15] += 1
            n_trig += 1
    pos = -(sig > 0).astype(float)  # short
    pnl = pos.shift(1) * moo
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K6 Barge surge short MOO")
    print_metrics(m)
    print(f"weeks={len(rate)}, n_trig={n_trig}, exposure_short={(pos<0).mean():.2%}")

    save_result("K6_grain_barge_rate", m, extra={
        "status": "ok",
        "rule": "Weekly ST LOUIS barge rate (% of tariff) WoW > +30% → short MOO 15 trading days from following Thu.",
        "mechanism": "Barge-rate surges signal river capacity stress that compresses agribusiness margins.",
        "source": "USDA AMS GTRFigure10Table9.xlsx, sheet TWK, ST LOUIS column (weekly % of tariff)",
        "weeks_observed": int(len(rate)),
        "n_triggers": int(n_trig),
        "exposure_pct": float((pos != 0).mean()),
    })


if __name__ == "__main__":
    main()
