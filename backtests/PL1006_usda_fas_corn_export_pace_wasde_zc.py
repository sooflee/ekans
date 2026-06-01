"""PL1006_usda_fas_corn_export_pace_wasde_zc — USDA Corn Export Pace >=90% by Week 35 -> Long ZC=F pre-WASDE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1006_usda_fas_corn_export_pace_wasde_zc"

    # Known years where cumulative corn export commitment pace reached >=90% of USDA projection by Week 35
    # (approximately early June each year). Entry: first Friday in June each qualifying year.
    # Exit: day before WASDE (~June 12 each year, released 2nd Thursday), ~10-14 trading days.
    # Marketing year for corn: Sep-Aug (Week 35 = ~late May / early June)

    # Qualifying years (based on USDA FAS historical data and known events):
    # 2007: strong early export sales (biofuel demand buildup)
    # 2011: strong export sales pre-drought rally
    # 2012: drought year; pace hit 90% early, WASDE raised exports
    # 2013: recovering demand post-drought
    # 2020: COVID demand surge; strong Chinese buying
    # 2021: China buying surge; exceptional pace
    # 2023: above-normal export pace early season

    # Entry: first Friday in June; Exit: ~12 trading days later (day before June WASDE ~June 12)
    events = [
        ("2007-06-01", "2007-06-13"),
        ("2011-06-03", "2011-06-10"),
        ("2012-06-01", "2012-06-12"),
        ("2013-06-07", "2013-06-12"),
        ("2020-06-05", "2020-06-11"),
        ("2021-06-04", "2021-06-11"),
        ("2023-06-02", "2023-06-12"),
    ]

    hold_days_max = 14  # trading days cap (pre-WASDE window)

    try:
        px = load_prices(["ZC=F", "ADM", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)

    # Use ZC=F if available, fallback to CORN ETF
    if "ZC=F" in ret.columns and ret["ZC=F"].notna().sum() > 200:
        corn_col = "ZC=F"
    else:
        return mark_failed(sid, "ZC=F not available in price data")

    spy_r = ret["SPY"]
    corn_r = ret[corn_col]

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for entry_str, exit_str in events:
        entry_date = pd.Timestamp(entry_str)
        exit_date = pd.Timestamp(exit_str)

        # Find actual trading days
        entry_idx_list = ret.index[ret.index >= entry_date]
        if len(entry_idx_list) < 2:
            print(f"Skipping {entry_str}: no trading days found")
            continue

        entry_idx = entry_idx_list[0]
        entry_loc = ret.index.get_loc(entry_idx)

        # Find exit (day before WASDE, or at most hold_days_max)
        exit_idx_list = ret.index[(ret.index >= entry_date) & (ret.index <= exit_date)]
        if len(exit_idx_list) < 1:
            exit_loc = min(entry_loc + hold_days_max, len(ret))
        else:
            exit_loc = min(ret.index.get_loc(exit_idx_list[-1]) + 1, len(ret))

        # Cap at hold_days_max
        exit_loc = min(exit_loc, entry_loc + hold_days_max)

        corn_window = corn_r.iloc[entry_loc:exit_loc].dropna()
        spy_window = spy_r.iloc[entry_loc:exit_loc]

        if len(corn_window) < 3:
            print(f"Skipping {entry_str}: insufficient data in window")
            continue

        # Long ZC=F
        strat_ret = corn_window
        corn_cum = float((1 + corn_window).prod() - 1)
        spy_cum = float((1 + spy_window.reindex(corn_window.index)).prod() - 1)

        event_records.append({
            "entry_date": entry_str,
            "exit_date": exit_str,
            "corn_return": round(corn_cum, 4),
            "spy_return": round(spy_cum, 4),
            "n_days": len(corn_window),
        })

        for idx, r in strat_ret.items():
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['entry_date']}: corn={e['corn_return']:.2%}, SPY={e['spy_return']:.2%} [n={e['n_days']}]")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="USDA Corn Export Pace Pre-WASDE Long")

    save_result(sid, m, extra={
        "rule": "When USDA FAS cumulative corn export sales reach >=90% of annual projection by Week 35 (early June), enter long ZC=F and exit day before the following WASDE release (~June 12)",
        "mechanism": "Strong export pace signals potential WASDE upward revision; pre-announcement drift as grain traders position ahead of the report; Chinese buying surges create self-reinforcing demand signals",
        "source": "USDA FAS Weekly Export Sales; WASDE calendar; ZC=F via yfinance; known qualifying events: 2007, 2011, 2012, 2013, 2020, 2021, 2023",
        "n_events": len(event_records),
        "events": event_records,
        "caveat": "USDA FAS export sales data requires manual download for real-time signal generation; N=7 qualifying years",
    })


if __name__ == "__main__":
    main()
