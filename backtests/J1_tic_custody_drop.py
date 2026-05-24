"""
J1 TIC custody drop -> short TLT.

FRED WMTSEC = Marketable Treasuries Held in Custody for Foreign Official and International Accounts (weekly Wed).
(Originally specified WCFOL, but that series is discontinued / not available; WMTSEC is the active replacement.)
Compute 4-week % change of the weekly series (approx MoM).
When 4-week change < -1%, short TLT for next 21 trading days.
Compare to TLT buy-and-hold.
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
        df = load_fred("WMTSEC", start="2002-01-01")
    except Exception as e:
        return mark_failed("J1_tic_custody_drop", f"FRED WMTSEC load failed: {e}")
    if df.empty:
        return mark_failed("J1_tic_custody_drop", "Empty FRED load")

    s = df.iloc[:, 0].dropna()
    chg4 = s.pct_change(4)
    triggers = chg4[chg4 < -0.01]

    px = load_prices(["TLT"], start="2002-01-01")
    if px.empty:
        return mark_failed("J1_tic_custody_drop", "TLT load failed")
    tlt = px["TLT"].dropna()
    rets = tlt.pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    for d in triggers.index:
        # find first trading day after this Wed
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 21, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = -1.0  # additive avoided: use last-write-wins
        n_events += 1

    pnl = (daily_pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J1 TIC custody drop -> short TLT")
    print_metrics(m)
    print(f"\nTrigger weeks: {n_events}")

    save_result("J1_tic_custody_drop", m, extra={
        "status": "ok",
        "rule": "FRED WMTSEC 4-week % change < -1% -> short TLT for 21 trading days.",
        "mechanism": "Foreign official custody outflows signal weaker overseas demand for USTs, pressuring long-bond prices.",
        "source": "https://fred.stlouisfed.org/series/WMTSEC (specified WCFOL is discontinued)",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
