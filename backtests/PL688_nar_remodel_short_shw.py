"""PL688 — NAR Existing Sales 4mo Drop + NAHB Remodel Roll → Short SHW, Long XHB"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL688_nar_remodel_short_shw"

    # Load housing starts from FRED as proxy for housing activity (EXHOSLUSM495S no longer has full history)
    # HOUST = Housing Starts Total, monthly since 1959
    try:
        houst = load_fred("HOUST", start="2010-01-01")
        houst = houst.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED HOUST: {e}")

    if houst.empty:
        return mark_failed(sid, "no housing starts data")

    # Strategy: trigger when housing starts decline MoM for 4+ consecutive months
    # AND housing starts level is below its 12-month moving average (broad housing weakness)
    # This proxies for the NAR existing home sales + NAHB remodel weakness condition

    # Compute MoM changes
    houst_mom = houst.diff()

    # Find trigger dates: 4 consecutive months of MoM declines in housing starts
    triggers = []
    i = 3  # need at least 4 months history
    while i < len(houst_mom):
        # Check if months i-3, i-2, i-1, i are all negative MoM
        window = [houst_mom.iloc[i-3], houst_mom.iloc[i-2], houst_mom.iloc[i-1], houst_mom.iloc[i]]
        if all(not np.isnan(v) and v < 0 for v in window):
            # Additional filter: level below 12-month MA (deep housing downturn proxy)
            ma12 = houst.iloc[max(0, i-11):i+1].mean()
            if houst.iloc[i] < ma12:
                td = houst.index[i]
                # Enforce 1-year cooldown
                if not triggers or (td - triggers[-1]).days > 365:
                    triggers.append(td)
        i += 1

    print(f"Trigger events found: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["SHW", "XHB", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SHW" not in px.columns:
        return mark_failed(sid, "price data missing SHW")

    ret = daily_returns(px)
    shw_r = ret["SHW"]
    xhb_r = ret["XHB"]
    spy_r = ret["SPY"]

    hold = 60
    pnl = pd.Series(0.0, index=spy_r.index)
    events_detail = []

    for td in triggers:
        # Find entry date: first trading day on or after FRED release date
        # FRED data is typically released ~3-4 weeks after month-end, add buffer
        entry_target = td + pd.DateOffset(days=25)
        mask = spy_r.index >= entry_target
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))
        window_idx = spy_r.index[p:ep]

        # Avoid overlap
        already_active = pnl.loc[window_idx] != 0.0
        if already_active.any():
            continue

        # Short SHW, long XHB 50% hedge
        shw = shw_r.reindex(window_idx).fillna(0.0)
        xhb = xhb_r.reindex(window_idx).fillna(0.0)
        period_pnl = -shw.values + 0.5 * xhb.values
        pnl.loc[window_idx] = period_pnl

        shw_ret = float((1 + shw).prod() - 1)
        xhb_ret = float((1 + xhb).prod() - 1)
        strategy_ret = -shw_ret + 0.5 * xhb_ret
        sp_ret = float((1 + spy_r.reindex(window_idx).fillna(0.0)).prod() - 1)

        events_detail.append({
            "trigger_date": str(td.date()),
            "entry_date": str(ei.date()),
            "houst_level": round(float(houst.loc[td]), 0),
            "shw_return": round(shw_ret, 4),
            "xhb_return": round(xhb_ret, 4),
            "strategy_return": round(strategy_ret, 4),
            "spy_return": round(sp_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid non-overlapping events found")

    active_pnl = pnl[pnl != 0.0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NAR Decline + NAHB Remodel Roll → Short SHW")
    avg_ret = np.mean([e["strategy_return"] for e in events_detail])
    win_rate = np.mean([e["strategy_return"] > 0 for e in events_detail])

    save_result(sid, m, extra={
        "rule": "Short SHW for 60 days when housing starts (HOUST) decline MoM for 4 consecutive months and level is below 12-month MA (deep housing downturn); hedge 50% long XHB",
        "mechanism": "Prolonged housing construction decline reduces paint/coatings demand; SHW revenue heavily tied to new construction and home improvement; XHB hedge limits homebuilder sector exposure",
        "source": "FRED HOUST (housing starts); yfinance SHW, XHB, SPY",
        "n_events": len(events_detail),
        "avg_event_return": round(float(avg_ret), 4),
        "event_win_rate": round(float(win_rate), 4),
        "events": events_detail,
    })
    print(f"Done: {len(events_detail)} events, avg_ret={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
