"""PL714_csu_klotzbach_long_reins — CSU Klotzbach Upgrade -> Long RNR/RLI/AXS"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL714_csu_klotzbach_long_reins"
    tickers = ["RNR", "RLI", "AXS", "SPY"]
    basket = ["RNR", "RLI", "AXS"]

    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded CSU July/August forecast upgrade events (named-storm count up >=2 vs NOAA May)
    # Based on CSU seasonal forecast archive 2015-2025
    # Events where CSU significantly revised UP vs NOAA May baseline:
    # 2017: Very active season, CSU Aug upgrade
    # 2020: Record season, CSU July upgrade
    # 2024: CSU Aug upgrade (known_events from strategy)
    # 2022: CSU Aug upgraded to above-normal
    events = [
        ("2017-08-03", 45),
        ("2019-08-01", 45),
        ("2020-07-09", 45),
        ("2022-08-04", 45),
        ("2024-08-01", 45),
    ]

    all_dates = ret.index
    pnl = pd.Series(0.0, index=all_dates)
    n_events = 0

    for event_date_str, hold_days in events:
        event_date = pd.Timestamp(event_date_str)
        valid = all_dates[all_dates >= event_date]
        if len(valid) == 0:
            continue
        entry_date = valid[0]
        idx_start = all_dates.get_loc(entry_date)
        idx_end = min(idx_start + hold_days, len(all_dates))
        window = all_dates[idx_start:idx_end]

        # Equal-weight basket return
        basket_ret = ret[basket].reindex(window).fillna(0).mean(axis=1)
        pnl.loc[window] += basket_ret
        n_events += 1

    if n_events == 0:
        return mark_failed(sid, "no valid events found")

    m = compute_metrics(pnl, benchmark=spy_r, name="CSU Klotzbach Upgrade -> Long RNR/RLI/AXS")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "When CSU (Klotzbach) July/August Atlantic hurricane forecast revises named-storm count UP by >=2 vs NOAA May, go long equal-weight RNR+RLI+AXS for 45 trading days",
        "mechanism": "Reinsurers price in elevated loss expectations ahead of active seasons; higher premiums and reduced competition from capacity withdrawal boost near-term earnings expectations for quality reinsurers",
        "source": "CSU tropical weather & climate research forecast archive; NOAA CPC seasonal outlook",
        "events": [{"date": d, "hold_days": h} for d, h in events],
    })


if __name__ == "__main__":
    main()
