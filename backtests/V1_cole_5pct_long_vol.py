"""
V1 Cole 5% long-vol overlay
Rule:
  Default position = 1x VIXM (long vol).
  When |SPY 21-day return| >= 5%, scale VIXM exposure to 2x for the next 21
  trading days. Otherwise stay 1x VIXM.
  Compared against SPY buy-and-hold (and against pure VIXM).
Mechanism (Cole / Artemis): tail-risk / long-vol is the most expensive sleeve
in a portfolio, so most days you want only nominal exposure; you "double down"
only when realised vol has actually broken out (|21d return| >= 5%) because
those episodes cluster (vol clustering, GARCH-style).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    try:
        px = load_prices(["SPY", "VIXM"], start="2011-01-01")
    except Exception as e:
        return mark_failed("V1_cole_5pct_long_vol", f"data load: {e}")

    px = px.dropna()
    if px.empty or "VIXM" not in px.columns:
        return mark_failed("V1_cole_5pct_long_vol", "no VIXM data")

    spy = px["SPY"]
    vixm = px["VIXM"]
    spy_r = spy.pct_change()
    vix_r = vixm.pct_change()

    # |SPY 21d return| >= 5%
    r21 = spy.pct_change(21)
    trig = r21.abs() >= 0.05

    # Position: 1.0 default, 2.0 for 21 trading days after each trigger
    weight = pd.Series(1.0, index=px.index)
    last_trigger = -10_000
    for i, dt in enumerate(px.index):
        if bool(trig.iloc[i]):
            last_trigger = i
        if i - last_trigger < 21:
            weight.iloc[i] = 2.0
        else:
            weight.iloc[i] = 1.0

    # apply next-day return, no look-ahead
    pnl = (weight.shift(1) * vix_r).dropna()

    bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V1 Cole 5% long-vol overlay")
    # also pure 1x VIXM for context
    m_base = compute_metrics(vix_r.reindex(pnl.index).dropna(), benchmark=bench,
                             name="V1 VIXM 1x baseline")
    print_metrics(m)
    print_metrics(m_base)

    save_result("V1_cole_5pct_long_vol", m, extra={
        "status": "ok",
        "rule": "Default 1x VIXM. When |SPY 21d return| >= 5%, scale to 2x VIXM for next 21 trading days.",
        "mechanism": "Vol clustering: realised vol shocks (|21d return|>=5%) cluster, so doubling long-vol after the trigger captures the next leg of an expanding-vol regime.",
        "source": "Christopher Cole / Artemis Capital — 'Allegory of the Hawk and Serpent', YouTube interview round 2 (Phase 1V).",
        "baseline_1x_vixm": m_base,
    })


if __name__ == "__main__":
    main()
