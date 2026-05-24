"""
F3 IIJA / FHWA quarterly obligation reports.

FHWA publishes quarterly Obligation Status reports showing federal-aid highway
obligations. When incremental quarterly obligations exceed ~$15B, long an
equal-weight basket of MLM, VMC, EXP, SUM for 30 trading days.

NOTE: FHWA obligation data is fragmented across PDF reports on
www.fhwa.dot.gov/federalaid/stattab.cfm and not easily machine-readable. The
quarterly increments below are approximate, reconstructed from FHWA fiscal-year
obligation totals and Federal Aid Highway Program (FAHP) press tracking.
Use with caution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (quarter_end_date, incremental_quarterly_obligations_usd_billions).
# Approximate; pre-IIJA quarters use the historical ~$45B/year FAHP run-rate.
# IIJA enacted Nov 2021 with $40B+ supplemental over baseline; quarterly
# obligations ramped through FY22-FY24.
QUARTERS = [
    ("2021-03-31", 10.5),
    ("2021-06-30", 11.2),
    ("2021-09-30", 13.5),
    ("2021-12-31", 11.8),
    ("2022-03-31", 12.4),
    ("2022-06-30", 13.7),
    ("2022-09-30", 16.2),   # IIJA ramp; exceeds threshold
    ("2022-12-31", 13.1),
    ("2023-03-31", 14.6),
    ("2023-06-30", 15.9),   # exceeds
    ("2023-09-30", 18.4),   # exceeds (Q4 FY surge)
    ("2023-12-31", 13.8),
    ("2024-03-31", 15.3),   # exceeds
    ("2024-06-30", 16.7),   # exceeds
    ("2024-09-30", 17.9),   # exceeds
    ("2024-12-31", 14.2),
    ("2025-03-31", 15.6),   # exceeds
]
THRESHOLD = 15.0


def main():
    sid = "F3_iija_fhwa"
    try:
        # Identify quarters that exceed threshold; signal is "next session after
        # quarter-end" since reports are published 30-60 days after.
        # We'll use quarter-end + 30 calendar days as the trade date proxy
        # (when reports become public).
        trigger_dates = []
        for q, val in QUARTERS:
            if val > THRESHOLD:
                qd = pd.Timestamp(q) + pd.Timedelta(days=45)
                trigger_dates.append(qd)

        if not trigger_dates:
            return mark_failed(sid, "No quarters above threshold")

        tickers = ["MLM", "VMC", "EXP", "SUM"]
        px = load_prices(tickers + ["SPY"], start="2020-06-01")
        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=tickers)
        used = []
        for D in trigger_dates:
            loc = idx.searchsorted(D, side="left")
            if loc >= len(idx):
                continue
            start = loc
            end = min(loc + 30, len(idx) - 1)
            for t in tickers:
                pos.iloc[start:end + 1, pos.columns.get_loc(t)] = 1.0 / len(tickers)
            used.append(str(D.date()))

        port = (pos.shift(1) * rets[tickers]).sum(axis=1)
        active = (pos.sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 10:
            return mark_failed(sid, "Too few active event days")

        spy_r = rets["SPY"]
        m = compute_metrics(port_active, benchmark=spy_r.reindex(port_active.index),
                            name="F3 IIJA/FHWA obligations -> aggregates basket")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok_approximate",
            "rule": "When quarterly incremental FHWA obligations > $15B, long equal-weight "
                    "MLM+VMC+EXP+SUM for 30 trading days starting ~45 days after quarter end.",
            "mechanism": "Materials demand pull-through from federal-aid highway obligations.",
            "universe": "MLM, VMC, EXP, SUM",
            "n_events": len(used),
            "events": used,
            "data_caveat": "Quarterly obligations are approximate reconstructions; FHWA "
                           "publishes obligation status only in fragmented PDF stat tabs.",
            "source": "FHWA Federal-Aid Obligation Status (stattab.cfm) — curated approximate",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
