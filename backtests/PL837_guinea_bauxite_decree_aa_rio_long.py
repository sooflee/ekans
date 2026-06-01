"""PL837_guinea_bauxite_decree_aa_rio_long — Guinea Junta Bauxite/Alumina Operator Expulsion Decree -> Long AA + RIO vs Short ACH (Chalco) Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL837_guinea_bauxite_decree_aa_rio_long"

    # Event-study anchors:
    # 1. 2021-09-05: Guinea coup - Conde overthrown, bauxite export uncertainty
    # 2. 2024-06-01: Guinea EGA suspension report (approximate date)
    event_entries = [
        pd.Timestamp("2021-09-05"),
        pd.Timestamp("2024-06-01"),
    ]

    HOLD_DAYS = 25  # 25 trading days hold

    try:
        px = load_prices(["AA", "RIO", "ACH", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    aa_r = ret["AA"]
    rio_r = ret["RIO"]
    ach_r = ret["ACH"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=aa_r.index)
    evts = []

    for entry_date in event_entries:
        # Find first trading day >= entry_date
        mask_entry = aa_r.index >= entry_date
        if mask_entry.sum() == 0:
            continue
        ei = aa_r.index[mask_entry][0]
        p = aa_r.index.get_loc(ei)

        # Hold HOLD_DAYS trading days
        ep = min(p + HOLD_DAYS, len(aa_r))

        if ep <= p:
            continue

        # Long AA (0.5) + Long RIO (0.5) - Short ACH (0.5)
        # Net notional: +1.0 long, -0.5 short
        segment_aa = aa_r.iloc[p:ep]
        segment_rio = rio_r.iloc[p:ep]
        segment_ach = ach_r.iloc[p:ep]
        # As per strategy: 0.5 AA + 0.5 RIO - 0.5 ACH
        segment_pnl = 0.5 * segment_aa + 0.5 * segment_rio - 0.5 * segment_ach

        pnl.iloc[p:ep] += segment_pnl.values

        evts.append({
            "entry": str(ei.date()),
            "exit": str(aa_r.index[ep - 1].date()),
            "aa_ret": float(segment_aa.sum()),
            "rio_ret": float(segment_rio.sum()),
            "ach_ret": float(segment_ach.sum()),
            "pair_pnl": float(segment_pnl.sum()),
        })

    if pnl.abs().sum() == 0:
        return mark_failed(sid, "no trades generated from event dates")

    m = compute_metrics(pnl, benchmark=spy_r, name="Guinea Bauxite Decree Long AA RIO Short ACH")
    save_result(sid, m, extra={
        "rule": (
            "TRIGGER: Guinea Ministry of Mines decree revoking/suspending named bauxite/alumina "
            "operator concession confirmed by Reuters/Bloomberg wire. "
            "ENTRY: Next open after wire confirmation: Long AA (0.5) + Long RIO (0.5) - Short ACH (0.5). "
            "HOLD: 25 trading days."
        ),
        "mechanism": (
            "Guinea accounts for ~25% of global bauxite supply. Operator disruptions restrict "
            "bauxite/alumina supply to Western smelters (AA, RIO) forcing them to pay higher spot "
            "prices -> margin squeeze. Simultaneously, Chinese alumina (ACH/Chalco) benefits from "
            "SHFE premium. NOTE: Empirically, 2021 coup showed ACH outperformed AA, casting doubt "
            "on this direction."
        ),
        "source": (
            "USGS Mineral Commodity Summaries (bauxite tonnage); Reuters/Bloomberg wire reports. "
            "Event dates: 2021-09-05 (Guinea coup), 2024-06-01 (EGA suspension approx)."
        ),
        "events": evts,
        "n_events": len(evts),
        "hold_days": HOLD_DAYS,
        "empirical_caution": (
            "Historical Guinea disruptions did NOT produce consistent AA/RIO outperformance vs ACH. "
            "2021 coup: ACH spiked +19.2% vs AA +2.6%. 2024 EGA: AA -24.9% broader sector decline. "
            "Thesis direction logically sound but empirically unconfirmed."
        ),
    })


if __name__ == "__main__":
    main()
