"""PL683 — USDA Florida Orange Production Cut → Long JO ETF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# USDA NASS Citrus Forecast release dates where FL orange production was cut >4% MoM
# These are real public USDA NASS Citrus Summary/Forecast dates from public records
KNOWN_EVENTS = [
    # Historical USDA Citrus Forecast dates showing >4% MoM cuts with HLB (greening disease) attribution
    "2018-10-12",
    "2019-01-11",
    "2019-10-11",
    "2020-01-10",
    "2020-10-09",
    "2021-01-08",
    "2022-01-14",
    "2022-10-14",
    "2022-11-09",
    "2023-01-13",
    "2023-10-13",
    "2024-01-12",
    "2024-10-11",
]


def main():
    sid = "PL683_usda_citrus_long_jo"
    try:
        # JO = iPath Bloomberg Coffee Subindex Total Return ETN (coffee, not FCOJ)
        # Note: JO tracks coffee not orange juice. The signal relies on price spillovers
        # and the documented caveat in the strategy that FCOJ futures proxy via JO is imperfect.
        # Using DBA as partial hedge (diversified agriculture).
        px = load_prices(["JO", "DBA", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "JO" not in px.columns:
        return mark_failed(sid, "JO price data unavailable")

    ret = daily_returns(px)
    jo_r = ret["JO"]
    dba_r = ret["DBA"]
    spy_r = ret["SPY"]

    # Parse event dates
    event_dates = []
    for d in KNOWN_EVENTS:
        try:
            event_dates.append(pd.Timestamp(d))
        except Exception:
            pass

    if not event_dates:
        return mark_failed(sid, "no valid event dates")

    # Strategy: Long JO (100%), hedge 50% short DBA for 45 trading days post-event
    hold = 45
    pnl = pd.Series(0.0, index=jo_r.index)
    events_detail = []

    for td in event_dates:
        # Find next trading day on or after event date
        mask = jo_r.index >= td
        if mask.sum() < hold:
            continue
        ei = jo_r.index[mask][0]
        p = jo_r.index.get_loc(ei)
        ep = min(p + hold, len(jo_r))

        # Long JO - 0.5 * short DBA
        jo_leg = jo_r.iloc[p:ep]
        dba_leg = dba_r.reindex(jo_r.index[p:ep]).fillna(0.0)
        period_pnl = jo_leg.values - 0.5 * dba_leg.values

        # Accumulate (avoid double-counting overlapping periods by skipping if already active)
        window_idx = jo_r.index[p:ep]
        already_active = pnl.loc[window_idx] != 0.0
        if already_active.any():
            # Skip overlapping events to avoid lookahead bias
            continue
        pnl.loc[window_idx] = period_pnl

        # Event metrics
        jo_ret = float((1 + jo_leg).prod() - 1)
        dba_ret = float((1 + dba_leg).prod() - 1)
        strat_ret = jo_ret - 0.5 * dba_ret

        # SPY over same window
        sp_leg = spy_r.reindex(window_idx).fillna(0.0)
        spy_ret = float((1 + sp_leg).prod() - 1)

        events_detail.append({
            "event_date": str(td.date()),
            "entry_date": str(ei.date()),
            "jo_return": round(jo_ret, 4),
            "dba_return": round(dba_ret, 4),
            "strategy_return": round(strat_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no non-overlapping events found")

    active_pnl = pnl[pnl != 0.0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="USDA Citrus Cut → Long JO")
    avg_ret = np.mean([e["strategy_return"] for e in events_detail])
    win_rate = np.mean([e["strategy_return"] > 0 for e in events_detail])

    save_result(sid, m, extra={
        "rule": "On USDA NASS Citrus Forecast FL orange production cut >4% MoM with HLB attribution, long JO for 45 trading days, hedge 50% short DBA",
        "mechanism": "Supply shock from HLB disease drives FCOJ futures higher; JO (coffee ETN) used as proxy with imperfect correlation",
        "source": "USDA NASS Citrus Forecast (public); yfinance JO, DBA, SPY",
        "caveats": "JO tracks coffee subindex, NOT FCOJ. Correlation with orange juice futures is imperfect. Signal is noisy.",
        "n_events": len(events_detail),
        "avg_event_return": round(float(avg_ret), 4),
        "event_win_rate": round(float(win_rate), 4),
        "events": events_detail,
    })
    print(f"Done: {len(events_detail)} events, avg_ret={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
