"""PL738_irs_45x_safeharbor_long_fslr_short_run — IRS 45X Safe-Harbor -> Long FSLR Short RUN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL738_irs_45x_safeharbor_long_fslr_short_run"
    # Hand-coded IRS 45X domestic-content safe-harbor events
    events = [
        "2023-12-14",  # 45X proposed safe-harbor
        "2024-05-16",  # Final 45X guidance
    ]

    try:
        px = load_prices(["FSLR", "RUN", "SPY"], start="2020-01-01")
        r = daily_returns(px)
        fslr_r = r["FSLR"]
        run_r = r["RUN"]
        spy_r = r["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 45  # trading days
    pnl = pd.Series(0.0, index=fslr_r.index)
    event_details = []

    for event_date_str in events:
        event_dt = pd.Timestamp(event_date_str)
        # Find first trading day on or after event
        mask = fslr_r.index >= event_dt
        if mask.sum() < hold:
            continue
        ei = fslr_r.index[mask][0]
        p = fslr_r.index.get_loc(ei)
        ep = min(p + hold, len(fslr_r))

        # Long FSLR / Short RUN spread
        fslr_seg = fslr_r.iloc[p:ep]
        run_seg = run_r.reindex(fslr_seg.index).fillna(0)
        spread = fslr_seg.values - run_seg.values[:len(fslr_seg)]

        pnl.iloc[p:ep] = spread[:ep - p]

        # Event stats
        fslr_ret = float((1 + fslr_seg).prod() - 1)
        run_ret = float((1 + run_seg.iloc[:ep - p]).prod() - 1)
        spy_seg = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_seg).prod() - 1)
        event_details.append({
            "event_date": event_date_str,
            "entry_date": str(ei.date()),
            "fslr_return": round(fslr_ret, 4),
            "run_return": round(run_ret, 4),
            "spread_return": round(fslr_ret - run_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

    active = pnl[pnl != 0]
    print(f"Events: {len(event_details)}, active PnL days: {len(active)}")

    if len(event_details) == 0:
        return mark_failed(sid, "no valid events")

    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="IRS 45X Safe-Harbor Long FSLR / Short RUN")
    save_result(sid, m, extra={
        "rule": "On IRS 45X domestic-content safe-harbor publication: long FSLR / short RUN for 45 trading days",
        "mechanism": "FSLR benefits disproportionately from domestic-content safe-harbor vs RUN (more installer exposure)",
        "source": "Hand-coded Treasury/IRS 45X notices 2023-2025; yfinance",
        "n_events": len(event_details),
        "events": event_details,
    })
    print(f"Done: Sharpe={m.get('sharpe', '?'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
