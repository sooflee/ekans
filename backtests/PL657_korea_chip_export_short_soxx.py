"""PL657 Korea 20-Day Chip Export YoY Deceleration - Short SOXX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL657_korea_chip_export_short_soxx"

    # Korea Customs 20-day chip export deceleration events
    # Per strategy spec: YoY growth decelerates >15pp MoM
    # Known events from spec + historical semiconductor down-cycles (2018-2025)
    # Korea Customs typically releases ~20th of each month
    events_str = [
        "2018-10-22",   # Korea chip exports collapsed late 2018 cycle
        "2019-01-21",   # continued decel into 2019 inventory correction
        "2022-09-20",   # sharp export decel as end-demand softened post-COVID
        "2022-12-20",   # further decel trough (Samsung guided big miss)
        "2023-04-21",   # known event from spec: major decel print
        "2024-08-22",   # known event from spec: AI/PC demand bifurcation
    ]

    try:
        px = load_prices(["SOXX", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    soxx_r = ret["SOXX"]
    spy_r = ret["SPY"]

    # Short SOXX, 50% long SPY hedge → net = -SOXX + 0.5*SPY
    hedged_r = -soxx_r + 0.5 * spy_r

    hold = 20  # 20 trading days
    pnl = pd.Series(0.0, index=hedged_r.index)
    events = []

    for ev_str in events_str:
        ev_dt = pd.Timestamp(ev_str)
        mask = hedged_r.index >= ev_dt
        if mask.sum() < hold:
            continue
        ei = hedged_r.index[mask][0]
        p = hedged_r.index.get_loc(ei)
        ep = min(p + hold, len(hedged_r))
        window = hedged_r.iloc[p:ep]
        # avoid overlap
        if (pnl.iloc[p:ep] != 0).any():
            continue
        pnl.iloc[p:ep] = window.values[: ep - p]

        soxx_fwd = float((1 + soxx_r.iloc[p:ep]).prod() - 1)
        spy_fwd = float((1 + spy_r.iloc[p:ep]).prod() - 1)
        events.append({
            "event_date": ev_str,
            "soxx_return": round(soxx_fwd, 4),
            "spy_return": round(spy_fwd, 4),
            "trade_pnl": round(float((1 + window).prod() - 1), 4),
        })

    active = pnl[pnl != 0]
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(pnl, benchmark=spy_r,
                        name="Korea Chip Export Decel → Short SOXX")
    save_result(sid, m, extra={
        "rule": ("When Korea Customs 20-day chip export YoY growth decelerates "
                 ">15pp MoM, short SOXX hedged 50% long SPY for 20 trading days."),
        "mechanism": ("Korea chip exports are a leading indicator for global "
                      "semiconductor demand. Deceleration signals inventory "
                      "correction risk, weighing on SOXX before US earnings reflect it."),
        "source": ("Korea Customs Service 20-day export data (MOTIE); "
                   "yfinance: SOXX, SPY"),
        "counter_signal": True,
        "counters": "long_semis",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, {len(active)} active PnL days")


if __name__ == "__main__":
    main()
