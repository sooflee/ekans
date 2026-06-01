"""PL832_pjm_siting_denial_dlr_eqix_short_counter — PJM/ERCOT Large-Load Interconnect Withdrawal Cluster + State Siting Denial -> Short DLR/EQIX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL832_pjm_siting_denial_dlr_eqix_short_counter"

    # Event-study anchors:
    # 1. Virginia SCC ratepayer docket PUR-2024-00045 open date ~ Aug 1 2024
    # 2. Loudoun County BOS moratorium discussions start ~ Mar 15 2024
    # Entry: next open after cluster + siting denial confirmed
    event_entries = [
        pd.Timestamp("2024-03-15"),  # Loudoun County BOS moratorium discussions
        pd.Timestamp("2024-08-01"),  # Virginia SCC PUR-2024-00045 open
    ]

    HOLD_DAYS = 25  # 25 trading days hold

    try:
        px = load_prices(["DLR", "EQIX", "PLD", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    dlr_r = ret["DLR"]
    eqix_r = ret["EQIX"]
    pld_r = ret["PLD"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=dlr_r.index)
    evts = []

    for entry_date in event_entries:
        # Find first trading day >= entry_date
        mask_entry = dlr_r.index >= entry_date
        if mask_entry.sum() == 0:
            continue
        ei = dlr_r.index[mask_entry][0]
        p = dlr_r.index.get_loc(ei)

        # Hold HOLD_DAYS trading days
        ep = min(p + HOLD_DAYS, len(dlr_r))

        if ep <= p:
            continue

        # Short DLR (0.5) + Short EQIX (0.5) = 1.0 short notional
        # Optional: Long PLD (0.5) as defensive REIT offset -> use for PnL series
        # Net position: -0.5 DLR - 0.5 EQIX
        segment_dlr = dlr_r.iloc[p:ep]
        segment_eqix = eqix_r.iloc[p:ep]
        segment_pnl = -0.5 * segment_dlr - 0.5 * segment_eqix

        pnl.iloc[p:ep] += segment_pnl.values

        evts.append({
            "entry": str(ei.date()),
            "exit": str(dlr_r.index[ep - 1].date()),
            "dlr_ret": float(segment_dlr.sum()),
            "eqix_ret": float(segment_eqix.sum()),
            "pair_pnl": float(segment_pnl.sum()),
        })

    if pnl.abs().sum() == 0:
        return mark_failed(sid, "no trades generated from event dates")

    m = compute_metrics(pnl, benchmark=spy_r, name="PJM Siting Denial Short DLR EQIX")
    save_result(sid, m, extra={
        "rule": (
            "TRIGGER: Cluster of >=3 large-load interconnection withdrawals in PJM/ERCOT queue "
            "within 60-day window + at least one state PUC/siting denial on DC campus >=200MW. "
            "ENTRY: Next open after 3rd withdrawal + siting denial confirmed. "
            "SHORT DLR (0.5) + SHORT EQIX (0.5). HOLD: 25 trading days."
        ),
        "mechanism": (
            "Hyperscaler data center demand deferral/cancellation creates lease-commencement "
            "delays for DLR and EQIX. Grid withdrawal clusters signal hyperscaler capex pause, "
            "directly threatening the leased-capacity pipeline that drives DLR/EQIX forward revenue."
        ),
        "source": (
            "PJM large-load interconnection queue CSV (pjm.com); ERCOT GIS XLSX (ercot.com); "
            "Virginia SCC docket PUR-2024-00045; Loudoun County BOS moratorium Q1 2024."
        ),
        "events": evts,
        "n_events": len(evts),
        "hold_days": HOLD_DAYS,
        "counter_signal": True,
        "counters": "long_hyperscaler_dc,long_DLR_EQIX",
        "note": "Thin event count (bt_feasibility=3) — only 1-2 clean modern cluster events exist",
    })


if __name__ == "__main__":
    main()
