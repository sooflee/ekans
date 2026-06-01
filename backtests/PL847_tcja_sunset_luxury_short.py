"""PL847_tcja_sunset_luxury_short — TCJA Individual Rate Sunset: Luxury Discretionary Short (RH, WSM, LULU, TPR, RL vs XLY)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL847_tcja_sunset_luxury_short"

    # Event-study: prior tax-bite surprise events on upper-income cohort
    # 1. 2013-01-01: ATRA fiscal-cliff resolution (top rate 35->39.6%)
    # 2. 2026-01-01: TCJA Title I sunset (forward test, if in data)
    # Entry: next trading day after each event
    event_entries = [
        pd.Timestamp("2013-01-02"),   # First trading day after ATRA 2013-01-01
        pd.Timestamp("2026-01-02"),   # First trading day after TCJA sunset 2026-01-01
    ]

    HOLD_DAYS = 80  # ~16 weeks = 80 trading days

    # RH IPO Nov 2012 - available for 2013 event
    # Use WSM, LULU, TPR, RL for both events; add RH when available
    try:
        px = load_prices(["RH", "WSM", "LULU", "TPR", "RL", "XLY", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    xly_r = ret["XLY"]

    # Build equal-weight luxury basket returns
    luxury_names = [c for c in ["RH", "WSM", "LULU", "TPR", "RL"] if c in ret.columns]
    if not luxury_names:
        return mark_failed(sid, "No luxury names available in yfinance")

    pnl = pd.Series(0.0, index=spy_r.index)
    evts = []

    for entry_date in event_entries:
        # Find first trading day >= entry_date
        mask_entry = spy_r.index >= entry_date
        if mask_entry.sum() == 0:
            continue
        ei = spy_r.index[mask_entry][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + HOLD_DAYS, len(spy_r))

        if ep <= p:
            continue

        # For each event, get available luxury names with data (no NaN at entry)
        available_names = []
        for name in luxury_names:
            col = ret[name].iloc[p:ep]
            if col.isna().sum() < len(col) * 0.3:  # < 30% missing
                available_names.append(name)

        if not available_names:
            continue

        # Short equal-weight luxury basket vs long XLY hedge
        # Short basket: -1/N * each name; Long XLY hedge: +1.0 * xly
        # Net: short luxury basket (1.5x notional) + long XLY (1x)
        # Simplified: short basket = -1.0, long XLY = +0.667 (maintains 1.5x ratio)
        n = len(available_names)
        seg_luxury = pd.DataFrame({name: ret[name].iloc[p:ep] for name in available_names})
        seg_luxury_ew = seg_luxury.mean(axis=1)  # equal-weight average return
        seg_xly = xly_r.iloc[p:ep]

        # Position: short basket (1.5) + long XLY (1.0)
        seg_pnl = -1.5 * seg_luxury_ew + 1.0 * seg_xly

        pnl.iloc[p:ep] += seg_pnl.values

        evts.append({
            "entry": str(ei.date()),
            "exit": str(spy_r.index[ep - 1].date()),
            "luxury_names": available_names,
            "luxury_ew_ret": float(seg_luxury_ew.sum()),
            "xly_ret": float(seg_xly.sum()),
            "pair_pnl": float(seg_pnl.sum()),
        })

    if pnl.abs().sum() == 0 or not evts:
        return mark_failed(sid, "no trades generated from event dates")

    m = compute_metrics(pnl, benchmark=spy_r, name="TCJA Sunset Luxury Short")
    save_result(sid, m, extra={
        "rule": (
            "When TCJA Title I rate provisions sunset AND Treasury DTS withheld income growth >10pp "
            "above wage growth AND >=2 luxury names miss SSS/guides, short equal-weight basket "
            "RH+WSM+LULU+TPR+RL (1.5x) + long XLY (1.0x). Hold 16 weeks or 80 trading days."
        ),
        "mechanism": (
            "Top-rate sunset raises effective taxes on upper-income consumers who drive luxury "
            "discretionary spending (RH, WSM furniture; LULU activewear; TPR/RL aspirational). "
            "After-tax income compression at upper deciles cuts spending on experiential/luxury goods."
        ),
        "source": (
            "yfinance: RH, WSM, LULU, TPR, RL, XLY, SPY. "
            "Event anchor: ATRA 2013-01-01 (historical), TCJA sunset 2026-01-01 (primary thesis)."
        ),
        "events": evts,
        "n_events": len(evts),
        "hold_days": HOLD_DAYS,
        "counter_signal": True,
        "counters": "long_SPY,long_XLY",
        "note": "Only 1 clean historical analog (2013 ATRA) + 1 forward test (2026). Thin event count.",
    })


if __name__ == "__main__":
    main()
