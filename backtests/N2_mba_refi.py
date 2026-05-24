"""
N2 MBA Refi proxy.
MBA Refi Index isn't free on FRED. Substitute: use FRED MORTGAGE30US weekly change.
Rule: Long REM (mortgage REIT ETF) 10 trading days when 30Y mortgage rate falls > 25 bp
WoW (proxy for refi-app spike).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        mtg = load_fred("MORTGAGE30US", start="2007-01-01").iloc[:, 0].rename("MTG")
        rem = load_prices(["REM"], start="2007-01-01").iloc[:, 0].rename("REM")
    except Exception as e:
        return mark_failed("N2_mba_refi", f"data load failed: {e}")

    mtg_wow = mtg.diff()  # WoW change in pct points
    trig = (mtg_wow < -0.25)  # drop > 25 bp
    trig_dates = mtg_wow.index[trig]

    rem_rets = rem.pct_change()
    pos = pd.Series(0.0, index=rem_rets.index)
    n_events = 0
    for d in trig_dates:
        loc = rem_rets.index.searchsorted(d)
        # mortgage rate publishes Thursday; trade next session for 10 days
        applied = False
        for k in range(1, 11):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0
                applied = True
        if applied:
            n_events += 1

    pnl_full = (pos * rem_rets).dropna()
    pnl = pnl_full.loc[pnl_full.ne(0).cummax()]
    if len(pnl) < 30:
        return mark_failed("N2_mba_refi", f"too few obs after trim: {len(pnl)}",
                           extra={"n_events": n_events})
    bench = rem_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N2 MBA refi (MORTGAGE30US proxy)")
    m["n_events"] = int(n_events)
    print(f"Trigger weeks: {n_events}")
    print_metrics(m)
    save_result("N2_mba_refi", m, extra={
        "status": "ok",
        "rule": "Long REM 10 sessions when MORTGAGE30US falls > 25bp WoW (refi-app spike proxy).",
        "mechanism": "Lower mortgage rates -> refi wave -> mortgage REIT NIM/prepay dynamics + sentiment",
        "universe": "REM",
        "source": "FRED MORTGAGE30US (proxy for MBA Refi Index, which is paywalled)",
        "data_substitution": "MBA Refi Index unavailable for free; using 30Y mortgage rate WoW change as proxy.",
    })


if __name__ == "__main__":
    main()
