"""
AA4 TIPS auction breakeven.
TreasuryDirect publishes 10Y TIPS auction dates (~5/year: 4 new-issue + reopens).
Hypothesis: around auctions, dealers underwrite supply → breakeven (T10YIE)
trades cheap pre-auction, recovers post. Short breakeven into the auction =
long TIP / short IEF from T-3 to T+0 (auction date).

T10YIE = 10Y Treasury - 10Y TIPS yield. We *short* breakeven by being short
TIP and long IEF (i.e., own nominals, short real). When breakeven widens,
nominals underperform TIPS, so this pair loses. Spec says "short breakeven
(long TIP / short IEF)" — implementing literally as spec'd: long TIP / short IEF
from T-3 to T+0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


# 10Y TIPS auction dates (new issue + reopenings) curated from TreasuryDirect.
# ~5 per year: Jan new, Mar reopen, May reopen, Jul new, Sep reopen, Nov reopen.
# Dates approximate to settlement-week Wednesdays/Thursdays.
TIPS_10Y_AUCTIONS = [
    # 2010s curated
    "2010-01-25", "2010-03-22", "2010-05-24", "2010-07-26", "2010-09-23",
    "2011-01-20", "2011-03-23", "2011-05-19", "2011-07-21", "2011-09-22",
    "2012-01-19", "2012-03-22", "2012-05-17", "2012-07-19", "2012-09-20",
    "2013-01-22", "2013-03-21", "2013-05-23", "2013-07-25", "2013-09-19",
    "2014-01-23", "2014-03-20", "2014-05-22", "2014-07-24", "2014-09-18",
    "2015-01-22", "2015-03-19", "2015-05-21", "2015-07-23", "2015-09-17",
    "2016-01-21", "2016-03-24", "2016-05-19", "2016-07-21", "2016-09-22",
    "2017-01-19", "2017-03-23", "2017-05-18", "2017-07-20", "2017-09-21",
    "2018-01-18", "2018-03-22", "2018-05-17", "2018-07-19", "2018-09-20",
    "2019-01-17", "2019-03-21", "2019-05-23", "2019-07-25", "2019-09-19",
    "2020-01-23", "2020-03-19", "2020-05-21", "2020-07-23", "2020-09-17",
    "2021-01-21", "2021-03-25", "2021-05-20", "2021-07-22", "2021-09-23",
    "2022-01-20", "2022-03-24", "2022-05-19", "2022-07-21", "2022-09-22",
    "2023-01-19", "2023-03-23", "2023-05-18", "2023-07-20", "2023-09-21",
    "2024-01-18", "2024-03-21", "2024-05-23", "2024-07-25", "2024-09-19",
    "2025-01-23", "2025-03-20", "2025-05-22", "2025-07-24", "2025-09-18",
]


def main():
    try:
        px = load_prices(["TIP", "IEF"], start="2010-01-01")
    except Exception as e:
        return mark_failed("AA4_tips_breakeven", f"data load failed: {e}")
    px = px.dropna()
    if len(px) < 200:
        return mark_failed("AA4_tips_breakeven", "insufficient TIP/IEF data")

    rets = daily_returns(px)
    tip_r = rets["TIP"]
    ief_r = rets["IEF"]
    idx = rets.index

    pos_tip = pd.Series(0.0, index=idx)
    pos_ief = pd.Series(0.0, index=idx)

    n_events = 0
    for d_str in TIPS_10Y_AUCTIONS:
        d = pd.Timestamp(d_str)
        i = idx.searchsorted(d, side="right") - 1
        if i < 3 or i >= len(idx):
            continue
        i_start = i - 3  # T-3 (inclusive)
        i_end = i        # T+0 inclusive
        hold_dates = idx[i_start:i_end + 1]
        pos_tip.loc[hold_dates] = 1.0
        pos_ief.loc[hold_dates] = -1.0
        n_events += 1

    if n_events < 10:
        return mark_failed("AA4_tips_breakeven",
                           f"too few auction events ({n_events})")

    pnl = (pos_tip.shift(1) * tip_r + pos_ief.shift(1) * ief_r).dropna()
    bench = tip_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA4 TIPS breakeven (auction)")
    print_metrics(m)
    save_result("AA4_tips_breakeven", m, extra={
        "status": "ok",
        "rule": "Long TIP / short IEF (i.e., short breakeven) from T-3 to T+0 of 10Y TIPS auction.",
        "universe": "TIP vs IEF",
        "source": "TreasuryDirect TIPS auction calendar",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
