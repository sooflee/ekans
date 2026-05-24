"""
I-10 Containerboard / paper-industry recession pulse.

Spec asked for AF&PA monthly containerboard production. AF&PA does not
publish a free time-series feed (members-only). We proxy with FRED:
   IPG322S = "Industrial Production: Paper Industry (NAICS 322)"
This is the cleanest free monthly substitute and tracks containerboard
demand fairly closely.

Rule:
  - 6-month YoY change of IPG322S.
  - When that YoY < -1.0 sigma of its trailing 10-year history, short IWM
    (Russell 2000 small-caps); hold until YoY turns positive.
  - Otherwise hold IWM long.
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
        df = load_fred("IPG322S", start="1995-01-01")
    except Exception as e:
        return mark_failed("I10_containerboard_pulse",
                           f"FRED IPG322S load failed: {e}")

    if df.empty:
        return mark_failed("I10_containerboard_pulse", "Empty FRED load")

    paper = df.iloc[:, 0].dropna()
    # 6-month YoY = (this month) / (6m ago) - 1 ? The prompt says "6-month YoY",
    # interpreting as "6-month change vs 6-months YoY" => use 12m % change of
    # the 6-month moving sum (i.e., compare a 6-month window to the prior
    # year's 6-month window). This smooths monthly noise.
    half_year_sum = paper.rolling(6).sum()
    yoy = half_year_sum.pct_change(12)

    # 10-year rolling z-score of YoY
    mu = yoy.rolling(120).mean()
    sigma = yoy.rolling(120).std()
    z = (yoy - mu) / sigma

    # Build monthly position: -1 (short IWM) when z < -1.0, +1 otherwise.
    pos_m = pd.Series(1.0, index=yoy.index)
    state = 1
    for d in yoy.index:
        if pd.isna(z.loc[d]):
            pos_m.loc[d] = 1.0
            continue
        if state == 1 and z.loc[d] < -1.0:
            state = -1
        elif state == -1 and yoy.loc[d] > 0:
            state = 1
        pos_m.loc[d] = float(state)

    # Trade IWM
    px = load_prices(["IWM"], start="1995-01-01")
    if px.empty or "IWM" not in px.columns:
        return mark_failed("I10_containerboard_pulse", "IWM load failed")

    iwm = px["IWM"].dropna()
    rets = iwm.pct_change()

    # Forward-fill the monthly position onto daily grid and lag 1 day
    daily_pos = pos_m.reindex(rets.index, method="ffill").fillna(1.0)
    pnl = (daily_pos.shift(1) * rets).dropna()

    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name="I-10 Paper-industry pulse -> IWM")
    print_metrics(m)
    n_short = int((pos_m < 0).sum())
    n_long = int((pos_m > 0).sum())
    print(f"\nMonthly state: long={n_long}, short={n_short}")

    save_result("I10_containerboard_pulse", m, extra={
        "status": "ok",
        "rule": ("6-month sum of FRED IPG322S YoY; when 10-yr rolling z-score "
                 "< -1.0, short IWM until YoY turns positive; else long IWM."),
        "data_source": "FRED IPG322S (Paper-industry IP; proxy for AF&PA "
                       "containerboard production which isn't publicly fed).",
        "n_months_short": n_short,
        "n_months_long": n_long,
        "caveats": ("IPG322S is broader than containerboard; substitutes for "
                    "lack of free AF&PA series."),
    })


if __name__ == "__main__":
    main()
