"""
P2 Karsan Post-OPEX.

Rule: Short SPY from Wednesday close of OPEX week through Tuesday close of
the week after monthly OPEX. (~4-5 trading days, the "post-OPEX vanna/charm
unwind" window where dealers reset and supply tends to swamp demand.)
"""
import sys
import calendar
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def third_friday(year, month):
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2] if len(fridays) >= 3 else None


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    n_windows = 0
    for y in range(idx[0].year, idx[-1].year + 1):
        for m in range(1, 13):
            tf = third_friday(y, m)
            if tf is None:
                continue
            opex = pd.Timestamp(tf)
            opex_wed = opex - pd.Timedelta(days=2)
            # Tuesday after OPEX Friday = opex + 4 calendar days
            post_tue = opex + pd.Timedelta(days=4)
            i_wed = idx.searchsorted(opex_wed, side="right") - 1
            i_tue = idx.searchsorted(post_tue, side="right") - 1
            if i_wed < 0 or i_tue < 0 or i_tue <= i_wed:
                continue
            hold_dates = idx[i_wed:i_tue]
            pos.loc[hold_dates] = -1.0
            n_windows += 1

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="P2 Karsan Post-OPEX (short)")
    print_metrics(m)
    save_result("P2_karsan_post_opex", m, extra={
        "status": "ok",
        "rule": "Short SPY from OPEX-Wednesday close through Tuesday-after-OPEX close (~4-5 trading days).",
        "mechanism": "Karsan: post-OPEX vanna/charm flow reverses; dealer hedges unwind, supply swamps demand.",
        "universe": "SPY",
        "n_windows": n_windows,
        "source": "Cem Karsan (YouTube/Twitter, Phase 1P)",
    })


if __name__ == "__main__":
    main()
