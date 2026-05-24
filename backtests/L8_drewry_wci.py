"""
L8 Freight surge proxy (Cass Freight Index from FRED) -> long XLI.

Drewry WCI weekly is behind a paywall/scraped chart; we use the Cass
Freight Shipments Index (FRGSHPUSM649NCIS) from FRED as a monthly proxy.

Rule:
- When Cass Freight Shipments Index has 3 consecutive months of > 2% MoM
  increases (proxy for the spec's "3 consec weeks >5%"), go long XLI for
  60 trading days from the trigger month-end.

Mechanism:
- Sustained freight throughput acceleration foreshadows industrial demand
  recovery; XLI (S&P 500 Industrials) benefits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        cass = load_fred(["FRGSHPUSM649NCIS"], start="2010-01-01")
    except Exception as e:
        return mark_failed("L8_drewry_wci", f"FRED Cass index fetch: {e}")
    s = cass["FRGSHPUSM649NCIS"].dropna()
    if s.empty:
        return mark_failed("L8_drewry_wci", "Cass series empty")

    mom = s.pct_change()
    # 3 consecutive months > 2% MoM
    cond = (mom > 0.02) & (mom.shift(1) > 0.02) & (mom.shift(2) > 0.02)
    trigger_dates = cond[cond].index

    px = load_prices(["XLI", "SPY"], start="2010-01-01")
    rets = px.pct_change()
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in trigger_dates:
        # publication lag ~3 weeks for Cass (approx) -> push entry by 21 calendar days
        eff = d + pd.Timedelta(days=21)
        nxt = rets.index[rets.index > eff]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        i = rets.index.get_loc(start)
        end = min(i + 60, len(rets.index))
        for j in range(i, end):
            pos.iloc[j] = 1.0
        last_end = rets.index[end - 1]
        n_events += 1

    pnl = (pos.shift(1) * rets["XLI"]).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="L8 Cass freight surge -> long XLI 60d")
    print_metrics(m)
    print(f"\nMonths: {len(s)} ; trigger months: {len(trigger_dates)} ; events: {n_events}")

    save_result("L8_drewry_wci", m, extra={
        "status": "ok",
        "rule": ("3 consecutive months Cass Freight Shipments Index MoM > 2% -> "
                 "long XLI for 60 trading days, with 21-day publication lag."),
        "mechanism": "Sustained freight acceleration foreshadows industrial demand recovery.",
        "source": "FRED FRGSHPUSM649NCIS (Cass Freight Index Shipments, monthly proxy for Drewry WCI weekly).",
        "n_events": int(n_events),
        "caveats": "Cass is monthly not weekly; Drewry WCI scraped data is paywalled.",
    })


if __name__ == "__main__":
    main()
