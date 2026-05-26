"""PL30_rig_count_production_lag — Rig Count -20% → Long Crude/XOP 6mo Later
After rig count drops >20% from peak, wait 6mo for DUC depletion, then long.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL30_rig_count_production_lag"
    try:
        px = load_prices(["XOP", "SPY"], start="2007-01-01")
        # CL=F can be tricky with roll; use XOP as primary
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xop_r = ret["XOP"]
    spy_r = ret["SPY"]

    # Hand-coded: entry dates = 6 months AFTER rig count crossed -20% from 12mo peak
    # Rig -20% crossing dates → +6mo entry:
    # Oct 2008 crossed → entry Apr 2009
    # Feb 2015 crossed → entry Aug 2015
    # May 2016 crossed (continued decline, -50%) → entry Nov 2016
    # Apr 2020 crossed → entry Nov 2020 (COVID rig collapse)
    # Oct 2023 crossed → entry Apr 2024
    entries = [
        ("2009-04-01", "GFC: rigs peaked 1600 Jul08, -20% by Oct08, +6mo entry"),
        ("2015-08-01", "Shale bust: rigs peaked 1609 Oct14, -20% by Feb15, +6mo"),
        ("2016-11-01", "Shale bust continued: -50% by May16, +6mo entry"),
        ("2020-11-01", "COVID: rigs peaked 683 Jan20, collapsed 65% by May20, +6mo"),
        ("2024-04-01", "2023 slowdown: rigs peaked 613 Jan23, -20% by Oct23, +6mo"),
    ]

    events = []
    pnl_parts = []
    hold_days = 126  # 6 months

    for date_str, desc in entries:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        xop_window = xop_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xop_window)

        xop_cum = float((1 + xop_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "entry_date": date_str,
            "description": desc,
            "xop_6m_return": round(xop_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(xop_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No events with data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Rig Count Lag → Long XOP")

    avg_xop = np.mean([e["xop_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xop_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "6mo after rig count crosses -20% from 12mo peak: long XOP for 6 months",
        "mechanism": "Rig decline → DUC depletion over 6mo → production stalls → supply tightens → oil/E&P rallies",
        "source": "Baker Hughes rig count (hand-coded crossing dates); yfinance XOP",
        "n_events": len(events),
        "avg_xop_return": round(avg_xop, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XOP: {avg_xop*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xop_6m_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']}: XOP {e['xop_6m_return']*100:+.1f}%, SPY {e['spy_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
