"""
V3 Felder margin-debt / GDP roll
Rule:
  Compute (margin debt / GDP), quarterly. Take 12-month change in that ratio.
  Default = 100% SPY.
  When the 12m-change in (margin/GDP) was > 2.5% of GDP and rolls negative
  (i.e. crosses from positive to negative), reduce SPY to 50%.
  Restore to 100% when the 12m change turns positive again.
Mechanism (Jesse Felder): peaks in (margin debt / GDP) historically coincide
with cycle tops; the inflection from rising to falling leverage signals the
unwind. The 2.5% threshold filters for periods of meaningful leverage build-up.
Data:
  Margin debt: FRED BOGZ1FL663067003Q (Broker/Dealer Margin Loans, quarterly).
  GDP:        FRED GDP (quarterly, nominal).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    try:
        md = load_fred("BOGZ1FL663067003Q", start="1980-01-01").iloc[:, 0].dropna().astype(float)
        gdp = load_fred("GDP", start="1980-01-01").iloc[:, 0].dropna().astype(float)
    except Exception as e:
        return mark_failed("V3_felder_margin_debt_gdp", f"FRED load: {e}")

    if len(md) < 30 or len(gdp) < 30:
        return mark_failed("V3_felder_margin_debt_gdp", "insufficient FRED history")

    # both quarterly — align to quarter-start
    df = pd.concat([md.rename("MD"), gdp.rename("GDP")], axis=1).dropna()
    df["ratio"] = df["MD"] / df["GDP"]
    df["chg_12m"] = df["ratio"].diff(4)  # 4 quarters = 12 months

    # State machine: 1 = full long, 0.5 = half.
    # Trigger to 0.5: chg_12m was > 0.025 in the last 4 quarters and now crosses
    # negative.
    state = 1.0
    pos_q = pd.Series(1.0, index=df.index)
    armed = False  # we recently saw chg_12m > 2.5%
    for i in range(len(df)):
        c = df["chg_12m"].iloc[i]
        if not np.isnan(c):
            if c > 0.025:
                armed = True
            if armed and c < 0 and state == 1.0:
                state = 0.5
                armed = False
            if state == 0.5 and c > 0:
                state = 1.0
        pos_q.iloc[i] = state

    # daily prices
    try:
        spy = load_prices(["SPY"], start="1993-01-29").iloc[:, 0]
    except Exception as e:
        return mark_failed("V3_felder_margin_debt_gdp", f"SPY load: {e}")
    rets = spy.pct_change()

    # forward-fill quarterly position to daily; lag one quarter day to avoid look-ahead
    daily_pos = pos_q.reindex(spy.index, method="ffill").fillna(1.0)
    pnl = (daily_pos.shift(1) * rets).dropna()
    # restrict to era we actually have FRED data for (post-1990)
    pnl = pnl.loc[pnl.index >= "1995-01-01"]
    m = compute_metrics(pnl, benchmark=rets.reindex(pnl.index),
                        name="V3 Felder margin-debt/GDP roll")
    print_metrics(m)
    save_result("V3_felder_margin_debt_gdp", m, extra={
        "status": "ok",
        "rule": "Reduce SPY to 50% when 12m chg of (margin/GDP) rolls from >2.5% positive to negative; back to 100% when 12m chg turns positive.",
        "mechanism": "Cycle-top diagnostic: peaks in leverage / GDP coincide with equity tops; the inflection (rising->falling) marks the unwind.",
        "source": "Jesse Felder, YouTube interview round 2 (Phase 1V).",
        "data_source": "FRED BOGZ1FL663067003Q (margin loans) / FRED GDP.",
    })


if __name__ == "__main__":
    main()
