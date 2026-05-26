"""PL42_semi_b2b_crossing_one_smh — SEMI B2B Crosses 1.0 from Below -> Long SMH
Hand-code months when SEMI NA B2B crossed above 1.0 from below (after 3+ months under).
Long SMH for 6 months. Events: 2009-07, 2013-05, 2016-08, 2019-11, 2024-01.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL42_semi_b2b_crossing_one_smh"
    try:
        px = load_prices(["SMH", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    smh_r = ret["SMH"]
    spy_r = ret["SPY"]

    # Hand-coded SEMI B2B crossing above 1.0 from below (after 3+ months under)
    crossings = [
        ("2009-07-15", "2009 GFC recovery — B2B crossed 1.0 after extreme downturn"),
        ("2013-05-15", "2013 — B2B recovery from 2012 memory downturn"),
        ("2016-08-15", "2016 — B2B recovery from 2015-16 industrial recession"),
        ("2019-11-15", "2019 — B2B recovery from 2018-19 memory correction"),
        ("2024-01-15", "2024 — B2B recovery from 2022-23 downcycle, AI demand"),
    ]

    events = []
    pnl_parts = []
    hold_days = 126  # 6 months

    for date_str, desc in crossings:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        if smh_r.iloc[window].isna().all():
            continue

        smh_window = smh_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(smh_window)

        smh_cum = float((1 + smh_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "entry_date": date_str,
            "description": desc,
            "smh_6m_return": round(smh_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(smh_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No B2B crossing events with SMH data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="SEMI B2B Cross 1.0 -> Long SMH")

    avg_smh = np.mean([e["smh_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["smh_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long SMH 6mo when SEMI B2B crosses 1.0 from below after 3+ months under",
        "mechanism": "B2B crossing 1.0 signals order book turning positive -> semiconductor sector recovery begins -> SMH rallies as earnings estimates inflect",
        "source": "SEMI B2B press releases (hand-coded crossing dates); yfinance SMH",
        "n_events": len(events),
        "avg_smh_return": round(avg_smh, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg SMH: {avg_smh*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["smh_6m_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']}: SMH {e['smh_6m_return']*100:+.1f}%, SPY {e['spy_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
