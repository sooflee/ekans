"""
J7 RRP drain -> short TLT.

FRED RRPONTSYD = Overnight Reverse Repo, daily.
When 5-day MA falls > 5% WoW (5-day pct change of the 5dMA), short TLT for next 5 trading days.
Spec phrased "> $50B WoW" -- we interpret that as ~5% on a $1T+ facility; we use a percent-change rule
so the test is regime-robust. (Series starts 2003 but RRP was tiny until 2021.)
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
        df = load_fred("RRPONTSYD", start="2014-01-01")
    except Exception as e:
        return mark_failed("J7_rrp_drain", f"FRED RRPONTSYD load failed: {e}")
    if df.empty:
        return mark_failed("J7_rrp_drain", "Empty FRED load")

    s = df.iloc[:, 0].dropna()
    # restrict to era when facility was material
    s = s[s.index >= "2021-04-01"]
    ma5 = s.rolling(5).mean()
    chg5 = ma5.diff(5)  # absolute change in $B
    # "drain of > $50B WoW" -> chg5 < -50
    trigger_dates = chg5[(chg5 < -50)].index

    px = load_prices(["TLT"], start="2020-01-01")
    if px.empty:
        return mark_failed("J7_rrp_drain", "TLT load failed")
    tlt = px["TLT"].dropna()
    rets = tlt.pct_change()

    daily_pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    for d in trigger_dates:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 5, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos.iloc[j] = -1.0
        n_events += 1

    pnl = (daily_pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="J7 RRP drain -> short TLT")
    print_metrics(m)
    print(f"\nTrigger days: {n_events}")

    save_result("J7_rrp_drain", m, extra={
        "status": "ok",
        "rule": "FRED RRPONTSYD 5d MA falls > $50B in 5d -> short TLT next 5 trading days.",
        "mechanism": "RRP drain indicates cash leaving money funds for T-bills/risk assets, often coincident with Treasury supply pressure on long bonds.",
        "source": "https://fred.stlouisfed.org/series/RRPONTSYD",
        "n_events": n_events,
        "caveats": "Sample restricted to 2021-present; RRP facility immaterial before then.",
    })


if __name__ == "__main__":
    main()
