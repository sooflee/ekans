"""
B03 VVIX/VIX ratio
When VVIX/VIX > 6.5 with VIX < 18, hedge (cash) for next 20 days.
When VVIX/VIX < 4.5, long SPY 20 days. Also report 1-year rolling z-score variant.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result, mark_failed, rolling_zscore,
)


def hold_window_positions(triggers, hold, sign=+1):
    """Given a boolean trigger series and hold period, build positions equal to `sign`
    for the next `hold` trading days after each trigger."""
    pos = pd.Series(0.0, index=triggers.index)
    remaining = 0
    for i in range(len(triggers)):
        if triggers.iloc[i]:
            remaining = hold
        if remaining > 0:
            pos.iloc[i] = sign
            remaining -= 1
    return pos


def main():
    try:
        vvix = load_prices(["^VVIX"], start="2007-01-01").iloc[:, 0].rename("VVIX")
        vix = load_prices(["^VIX"], start="2007-01-01").iloc[:, 0].rename("VIX")
        spy = load_prices(["SPY"], start="2007-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("B03_vvix_vix_ratio", f"data load failed: {e}")

    df = pd.concat([vvix, vix, spy], axis=1).dropna()
    if df.empty:
        return mark_failed("B03_vvix_vix_ratio", "no overlap between ^VVIX and ^VIX")
    rets = df["SPY"].pct_change()

    ratio = df["VVIX"] / df["VIX"]
    hi_trig = (ratio > 6.5) & (df["VIX"] < 18)
    lo_trig = (ratio < 4.5)

    # Combine: hi -> cash (0) for 20d; lo -> long (+1) for 20d; else neutral (=0)
    # Implementation: start from default 0; lo triggers a +1 window; hi triggers 0 window (already 0).
    # To distinguish: build long windows and a "hedge override".
    long_pos = hold_window_positions(lo_trig, 20, sign=+1)
    hedge_pos = hold_window_positions(hi_trig, 20, sign=+1)  # marks hedge window
    pos = long_pos.copy()
    pos[hedge_pos > 0] = 0.0  # hedge wins over long

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="B03 VVIX/VIX absolute thresholds")

    # ----- z-score variant -----
    z = rolling_zscore(ratio, 252)
    hi_z = z > 1.5
    lo_z = z < -1.5
    long_pos_z = hold_window_positions(lo_z, 20, sign=+1)
    hedge_pos_z = hold_window_positions(hi_z, 20, sign=+1)
    pos_z = long_pos_z.copy()
    pos_z[hedge_pos_z > 0] = 0.0
    pnl_z = long_short_pnl(pos_z, rets)
    m_z = compute_metrics(pnl_z, benchmark=rets, name="B03 VVIX/VIX z-score variant")
    print_metrics(m)
    print_metrics(m_z)

    save_result("B03_vvix_vix_ratio", m, extra={
        "status": "ok",
        "rule": "If VVIX/VIX>6.5 & VIX<18, cash 20d; if VVIX/VIX<4.5, long SPY 20d.",
        "variant_zscore": m_z,
        "universe": "SPY (signal: ^VVIX/^VIX)",
        "source": "Vol-of-vol literature; ratio interpretation",
    })


if __name__ == "__main__":
    main()
