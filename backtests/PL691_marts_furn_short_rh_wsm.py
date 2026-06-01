"""PL691 — Census MARTS Furniture YoY Collapse → Short RH/WSM, Long XHB"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL691_marts_furn_short_rh_wsm"

    # Load RSFHFSN = Retail Sales: Furniture and Home Furnishings (SA, monthly)
    try:
        furn = load_fred("RSFHFSN", start="2010-01-01")
        furn = furn.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED RSFHFSN: {e}")

    if furn.empty or len(furn) < 24:
        return mark_failed(sid, f"insufficient furniture data ({len(furn)} rows)")

    # Compute YoY
    furn_yoy = furn.pct_change(12)

    # Find trigger dates: 2 consecutive months of YoY < -6%
    threshold = -0.06
    triggers = []
    i = 1
    while i < len(furn_yoy):
        v_cur = furn_yoy.iloc[i]
        v_prev = furn_yoy.iloc[i-1]
        if (not np.isnan(v_cur) and v_cur < threshold and
                not np.isnan(v_prev) and v_prev < threshold):
            td = furn_yoy.index[i]
            # Enforce 6-month cooldown
            if not triggers or (td - triggers[-1]).days > 180:
                triggers.append(td)
        i += 1

    print(f"Trigger events found: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no trigger events with 2 consecutive months YoY < -6%")

    try:
        px = load_prices(["RH", "WSM", "XHB", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "price data empty")

    ret = daily_returns(px)
    rh_r = ret["RH"] if "RH" in ret.columns else None
    wsm_r = ret["WSM"] if "WSM" in ret.columns else None
    xhb_r = ret["XHB"] if "XHB" in ret.columns else None
    spy_r = ret["SPY"]

    if rh_r is None or wsm_r is None:
        return mark_failed(sid, "RH or WSM not available")

    hold = 45
    pnl = pd.Series(0.0, index=spy_r.index)
    events_detail = []

    for td in triggers:
        # MARTS data is released with ~4-6 week lag; data for month M released ~mid M+1
        # Add 45 days to account for release delay + look-ahead protection
        entry_target = td + pd.DateOffset(days=45)
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

        # Short RH and WSM equal-weight; hedge 50% long XHB
        rh = rh_r.reindex(window_idx).fillna(0.0)
        wsm = wsm_r.reindex(window_idx).fillna(0.0)
        short_basket = (rh.values + wsm.values) / 2  # equal-weight short
        xhb = xhb_r.reindex(window_idx).fillna(0.0) if xhb_r is not None else pd.Series(0.0, index=window_idx)
        period_pnl = -short_basket + 0.5 * xhb.values
        pnl.loc[window_idx] = period_pnl

        rh_ret = float((1 + rh).prod() - 1)
        wsm_ret = float((1 + wsm).prod() - 1)
        xhb_ret = float((1 + xhb).prod() - 1) if xhb_r is not None else 0.0
        short_basket_ret = (rh_ret + wsm_ret) / 2
        strategy_ret = -short_basket_ret + 0.5 * xhb_ret
        sp_ret = float((1 + spy_r.reindex(window_idx).fillna(0.0)).prod() - 1)
        yoy_val = float(furn_yoy.loc[td])

        events_detail.append({
            "trigger_date": str(td.date()),
            "entry_date": str(ei.date()),
            "furn_yoy": round(yoy_val, 4),
            "rh_return": round(rh_ret, 4),
            "wsm_return": round(wsm_ret, 4),
            "xhb_return": round(xhb_ret, 4),
            "strategy_return": round(strategy_ret, 4),
            "spy_return": round(sp_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid non-overlapping events found")

    active_pnl = pnl[pnl != 0.0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="MARTS Furn Collapse → Short RH/WSM")
    avg_ret = np.mean([e["strategy_return"] for e in events_detail])
    win_rate = np.mean([e["strategy_return"] > 0 for e in events_detail])

    save_result(sid, m, extra={
        "rule": "When RSFHFSN furniture retail YoY < -6% for 2 consecutive months, short RH and WSM equal-weight for 45 trading days, hedge 50% long XHB",
        "mechanism": "Furniture sales collapse signals discretionary spending weakness; RH and WSM are leveraged to high-end home furnishings demand which amplifies downside vs. broader housing ETF",
        "source": "FRED RSFHFSN (furniture retail sales); yfinance RH, WSM, XHB, SPY",
        "n_events": len(events_detail),
        "avg_event_return": round(float(avg_ret), 4),
        "event_win_rate": round(float(win_rate), 4),
        "events": events_detail,
    })
    print(f"Done: {len(events_detail)} events, avg_ret={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
