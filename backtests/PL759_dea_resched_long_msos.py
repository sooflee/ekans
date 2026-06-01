"""PL759_dea_resched_long_msos — DEA Federal Register Rescheduling Milestone -> Long MSOS"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL759_dea_resched_long_msos"

    # Hand-coded DEA/Federal Register cannabis rescheduling milestones
    # MSOS launched Sep 2020; milestones 2022-2025
    # Sources: DEA Federal Register, trade press
    events_raw = [
        # Biden directive to HHS/DEA to review scheduling: Oct 2022
        ("2022-10-06", "Biden cannabis scheduling review directive"),
        # HHS recommendation to DEA to reschedule to Schedule III: Aug 2023
        ("2023-08-29", "HHS recommends Schedule III rescheduling"),
        # DEA NPRM (Notice of Proposed Rulemaking) to reschedule to Sch III: Apr 2024
        ("2024-04-30", "DEA NPRM - proposed cannabis rescheduling to Schedule III"),
        # Federal Register public comment period opens: May 2024
        ("2024-05-21", "Federal Register NPRM public comment period"),
        # DEA scheduling hearing announcement: 2025
        ("2025-01-15", "DEA administrative hearing announcement"),
    ]

    try:
        px = load_prices(["MSOS", "SPY"], start="2020-10-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "MSOS" not in px.columns:
        return mark_failed(sid, "MSOS price data unavailable")

    ret = daily_returns(px)
    msos_r = ret["MSOS"].dropna()
    spy_r = ret["SPY"].dropna()

    if len(msos_r) < 45:
        return mark_failed(sid, f"insufficient MSOS data: {len(msos_r)} days")

    hold_days = 45  # 45 trading days
    pnl = pd.Series(0.0, index=msos_r.index)
    evts = []

    for event_str, label in events_raw:
        event_date = pd.Timestamp(event_str)

        # Find entry point: first trading day on or after the milestone date
        after_event = msos_r.index[msos_r.index >= event_date]
        if len(after_event) == 0:
            continue  # Event after available data

        entry_idx = after_event[0]
        entry_pos = msos_r.index.get_loc(entry_idx)

        # Exit: hold_days trading days later
        exit_pos = min(entry_pos + hold_days, len(msos_r))
        window = msos_r.iloc[entry_pos:exit_pos]

        if len(window) < 5:
            continue

        # Mark positions (avoid double-counting overlapping events)
        pnl.iloc[entry_pos:exit_pos] = window.values

        cum_ret = float((1 + window).prod() - 1)
        spy_window = spy_r.reindex(window.index)
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else None

        evts.append({
            "milestone": event_str,
            "label": label,
            "entry_date": str(entry_idx.date()),
            "n_days": len(window),
            "msos_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events processed: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    # Use only active-position days
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="DEA Rescheduling Milestones -> Long MSOS")
    returns_list = [e["msos_return"] for e in evts]

    save_result(sid, m, extra={
        "rule": "Long MSOS for 45 trading days starting on DEA/FR cannabis rescheduling milestone date",
        "mechanism": "Rescheduling milestones trigger re-rating of cannabis MSO equity: removal of 280E tax burden, banking access, institutional eligibility",
        "source": "DEA Federal Register milestones 2022-2025; yfinance MSOS, SPY",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")
    for e in evts:
        print(f"  {e['label']}: MSOS={e['msos_return']:.1%}, SPY={e.get('spy_return', 'N/A')}")


if __name__ == "__main__":
    main()
