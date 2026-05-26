"""PL10_class8_truck_replacement — Class 8 Order Trough → Long PCAR+CMI
12 months after Class 8 order trough, replacement cycle kicks in.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL10_class8_truck_replacement"
    try:
        px = load_prices(["PCAR", "CMI", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    basket = (ret["PCAR"] + ret["CMI"]) / 2
    spy_r = ret["SPY"]

    # Class 8 order troughs (ACT Research, hand-coded)
    # Entry = 12 months after trough, hold 12 months
    troughs = [
        ("2009-05-01", "GFC trough"),
        ("2016-01-01", "Industrial recession trough"),
        ("2019-10-01", "Freight recession trough"),
        ("2023-01-01", "Post-COVID normalization trough"),
    ]

    events = []
    pnl_parts = []
    hold_days = 252

    for trough_date_str, desc in troughs:
        entry_date = pd.Timestamp(trough_date_str) + pd.DateOffset(months=12)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        daily_pnl = basket.iloc[window]
        pnl_parts.append(daily_pnl)

        cum_ret = float((1 + daily_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.iloc[window]).prod() - 1)
        events.append({
            "trough": trough_date_str,
            "entry": str(ret.index[entry_loc].date()),
            "description": desc,
            "basket_12m_return": round(cum_ret, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(cum_ret - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No events")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Class 8 Replacement Cycle")
    save_result(sid, m, extra={
        "rule": "12 months after Class 8 order trough: long PCAR+CMI for 12 months",
        "mechanism": "Deferred truck replacement during downturn creates forced replacement cycle 18mo later",
        "source": "ACT Research Class 8 order data; yfinance",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
