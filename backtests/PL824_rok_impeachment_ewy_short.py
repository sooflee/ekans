"""PL824_rok_impeachment_ewy_short — ROK Presidential Approval Collapse + Impeachment Motion -> Short EWY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL824_rok_impeachment_ewy_short"
    # Three known impeachment event dates (entry day after trigger confirmation)
    event_entries = [
        pd.Timestamp("2004-03-12"),   # Roh Moo-hyun impeachment motion (dismissed May 14 2004)
        pd.Timestamp("2016-12-09"),   # Park Geun-hye impeachment motion (upheld Mar 10 2017)
        pd.Timestamp("2024-12-04"),   # Yoon Suk-yeol post martial-law (upheld Apr 4 2025)
    ]

    # Corresponding verdict / exit dates
    event_exits = [
        pd.Timestamp("2004-05-14"),   # Roh: dismissed ~63 trading days
        pd.Timestamp("2017-03-10"),   # Park: upheld ~63 trading days
        pd.Timestamp("2025-04-04"),   # Yoon: upheld ~85 trading days
    ]

    try:
        px = load_prices(["EWY", "SPY"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    ewy_r = ret["EWY"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=ewy_r.index)
    evts = []

    for entry_date, exit_date in zip(event_entries, event_exits):
        # Find entry index (first trading day >= entry_date)
        mask_entry = ewy_r.index >= entry_date
        if mask_entry.sum() == 0:
            continue
        ei = ewy_r.index[mask_entry][0]

        # Find exit index (first trading day >= exit_date)
        mask_exit = ewy_r.index >= exit_date
        if mask_exit.sum() == 0:
            # If verdict date is in the future, hold through end of data
            ep_idx = len(ewy_r)
        else:
            exit_i = ewy_r.index[mask_exit][0]
            ep_idx = ewy_r.index.get_loc(exit_i) + 1  # include the exit day

        p = ewy_r.index.get_loc(ei)
        ep = min(ep_idx, len(ewy_r))

        if ep <= p:
            continue

        # SHORT EWY: pnl = -1 * EWY daily returns
        er = -ewy_r.iloc[p:ep]
        pnl.iloc[p:ep] = er.values

        # Cumulative returns for this event
        ewy_cum = float((1 + ewy_r.iloc[p:ep]).prod() - 1)
        short_cum = float((1 + er).prod() - 1)

        spy_cum = None
        if ei in spy_r.index:
            sp = spy_r.index.get_loc(ei)
            spy_cum = float((1 + spy_r.iloc[sp:min(sp + (ep - p), len(spy_r))]).prod() - 1)

        evts.append({
            "entry_date": str(ei.date()),
            "exit_date": str(exit_date.date()),
            "ewy_return": round(ewy_cum, 4),
            "short_ewy_return": round(short_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events processed: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    print(f"Active PnL days: {len(ip)}")

    if len(ip) < 10:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="ROK Impeachment → Short EWY")
    m["n_events"] = len(evts)

    save_result(sid, m, extra={
        "rule": "Short EWY when ROK presidential approval <25% for 3 consecutive weeks AND impeachment motion tabled or Constitutional Court hearing scheduled. Hold through verdict.",
        "mechanism": "Political risk premium expansion in Korean equities during constitutional crisis. Foreign outflows from EWY ETF as institutional investors reduce exposure to governance uncertainty. KRW weakness amplifies losses for USD-based investors in EWY.",
        "source": "Public record: Roh 2004-03-12, Park 2016-12-09, Yoon 2024-12-04; EWY and SPY via yfinance",
        "n_events": len(evts),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, metrics saved.")
    for e in evts:
        print(f"  {e['entry_date']} -> {e['exit_date']}: short_ewy={e['short_ewy_return']:.1%}, spy={e['spy_return']:.1%}" if e['spy_return'] else f"  {e['entry_date']} -> {e['exit_date']}: short_ewy={e['short_ewy_return']:.1%}")


if __name__ == "__main__":
    main()
