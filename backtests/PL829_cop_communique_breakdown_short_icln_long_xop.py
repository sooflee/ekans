"""PL829_cop_communique_breakdown_short_icln_long_xop — COP Final-Day Communique Breakdown (Fossil Phase-Out Weakening) -> Short ICLN / Long XOP"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL829_cop_communique_breakdown_short_icln_long_xop"

    # Entry dates: first US trading day after each COP final gavel
    # where the final cover text weakened phase-out language vs published draft
    # COP26 Glasgow: gavel Nov 13 2021 (Sat) -> entry Mon Nov 15
    # COP27 Sharm el-Sheikh: gavel Nov 20 2022 (Sun) -> entry Mon Nov 21
    # COP28 Dubai: gavel Dec 13 2023 (Wed) -> entry Thu Dec 14
    # COP29 Baku: gavel Nov 24 2024 (Sun) -> entry Mon Nov 25
    event_entries = [
        pd.Timestamp("2021-11-15"),
        pd.Timestamp("2022-11-21"),
        pd.Timestamp("2023-12-14"),
        pd.Timestamp("2024-11-25"),
    ]

    HOLD_DAYS = 10  # 10 trading days hold

    try:
        px = load_prices(["ICLN", "XOP", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    icln_r = ret["ICLN"]
    xop_r = ret["XOP"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=icln_r.index)
    evts = []

    for entry_date in event_entries:
        # Find first trading day >= entry_date
        mask_entry = icln_r.index >= entry_date
        if mask_entry.sum() == 0:
            continue
        ei = icln_r.index[mask_entry][0]
        p = icln_r.index.get_loc(ei)

        # Hold HOLD_DAYS trading days
        ep = min(p + HOLD_DAYS, len(icln_r))

        if ep <= p:
            continue

        # Equal-notional: short ICLN, long XOP
        # Short ICLN: pnl = -icln_r; Long XOP: pnl = +xop_r
        # Each side is 50% of notional -> combined = 0.5 * (-icln_r + xop_r)
        segment_icln = icln_r.iloc[p:ep]
        segment_xop = xop_r.iloc[p:ep]
        segment_pnl = 0.5 * (-segment_icln + segment_xop)

        pnl.iloc[p:ep] += segment_pnl.values

        evts.append({
            "entry": str(ei.date()),
            "exit": str(icln_r.index[ep - 1].date()),
            "icln_ret": float(segment_icln.sum()),
            "xop_ret": float(segment_xop.sum()),
            "pair_pnl": float(segment_pnl.sum()),
        })

    if pnl.abs().sum() == 0:
        return mark_failed(sid, "no trades generated from event dates")

    # Align spy benchmark to non-zero pnl days
    m = compute_metrics(pnl, benchmark=spy_r, name="COP Communique Breakdown Short ICLN Long XOP")
    save_result(sid, m, extra={
        "rule": (
            "When UN COP climate conference extends >36h past scheduled close AND "
            "final cover decision weakens 'phase-out of fossil fuels' language vs published draft, "
            "enter SHORT ICLN + LONG XOP equal-notional at open on first US trading day after final gavel. "
            "Hold 10 trading days."
        ),
        "mechanism": (
            "Fossil-fuel phase-out language weakening signals regulatory/policy relief for oil sector "
            "and disappointment for clean energy investors. Market re-prices energy policy risk: "
            "XOP (oil & gas explorers) benefits from reduced transition pressure; "
            "ICLN (global clean energy) sells off on weakened policy tailwind."
        ),
        "source": (
            "UNFCCC website (unfccc.int) published draft vs final cover decisions. "
            "Event dates: COP26 Glasgow 2021-11-15, COP27 Sharm 2022-11-21, "
            "COP28 Dubai 2023-12-14, COP29 Baku 2024-11-25."
        ),
        "events": evts,
        "n_events": len(evts),
        "hold_days": HOLD_DAYS,
    })


if __name__ == "__main__":
    main()
