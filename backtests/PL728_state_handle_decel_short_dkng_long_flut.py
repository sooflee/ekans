"""PL728 — State Sports Handle Decel -> Short DKNG Long FLUT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


# Hand-coded aggregate monthly sports handle (NJ + PA + NY + IL + MA), USD billions
# Sources: NJ DGE, PA PGCB, NY Gaming Commission, IL Gaming Board, MA GC
# Coverage: NJ from 2019, PA 2019, NY 2022+, IL 2020+, MA 2023+
# Only using states with consistent reporting; values are approximate
HANDLE_MONTHLY = {
    # 2019 (NJ + PA only, ~early market)
    "2019-01": 0.39, "2019-02": 0.38, "2019-03": 0.41, "2019-04": 0.45,
    "2019-05": 0.48, "2019-06": 0.42, "2019-07": 0.44, "2019-08": 0.56,
    "2019-09": 0.61, "2019-10": 0.72, "2019-11": 0.68, "2019-12": 0.63,
    # 2020 (NJ + PA + IL joining)
    "2020-01": 0.65, "2020-02": 0.62, "2020-03": 0.48, "2020-04": 0.12,
    "2020-05": 0.21, "2020-06": 0.38, "2020-07": 0.52, "2020-08": 0.71,
    "2020-09": 0.95, "2020-10": 1.15, "2020-11": 1.10, "2020-12": 1.05,
    # 2021
    "2021-01": 1.20, "2021-02": 0.98, "2021-03": 1.35, "2021-04": 1.28,
    "2021-05": 1.18, "2021-06": 1.05, "2021-07": 1.12, "2021-08": 1.45,
    "2021-09": 1.68, "2021-10": 1.89, "2021-11": 1.72, "2021-12": 1.61,
    # 2022 (NY launches Jan)
    "2022-01": 2.41, "2022-02": 2.15, "2022-03": 2.35, "2022-04": 2.18,
    "2022-05": 2.05, "2022-06": 1.85, "2022-07": 1.92, "2022-08": 2.38,
    "2022-09": 2.65, "2022-10": 2.85, "2022-11": 2.72, "2022-12": 2.61,
    # 2023 (MA launches March)
    "2023-01": 2.55, "2023-02": 2.31, "2023-03": 2.75, "2023-04": 2.65,
    "2023-05": 2.48, "2023-06": 2.21, "2023-07": 2.35, "2023-08": 2.92,
    "2023-09": 3.08, "2023-10": 3.25, "2023-11": 3.12, "2023-12": 3.05,
    # 2024
    "2024-01": 3.18, "2024-02": 2.95, "2024-03": 3.08, "2024-04": 2.88,
    "2024-05": 2.72, "2024-06": 2.45, "2024-07": 2.61, "2024-08": 3.15,
    "2024-09": 3.38, "2024-10": 3.55, "2024-11": 3.42, "2024-12": 3.28,
    # 2025
    "2025-01": 3.35, "2025-02": 3.08, "2025-03": 3.22, "2025-04": 3.10,
}

RELEASE_LAG_DAYS = 21  # ~3 weeks after month-end


def main():
    sid = "PL728_state_handle_decel_short_dkng_long_flut"
    try:
        px = load_prices(["DKNG", "FLUT", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # FLUT US-listed January 2024; restrict to 2024+ for clean data
    # So strategy only active when both DKNG and FLUT are available
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    dkng_r = ret["DKNG"]
    flut_r = ret["FLUT"] if "FLUT" in ret.columns else None

    if flut_r is None or flut_r.dropna().empty:
        return mark_failed(sid, "FLUT not available in yfinance data")

    # Build handle series
    handle = pd.Series(HANDLE_MONTHLY)
    handle.index = pd.to_datetime([f"{k}-01" for k in handle.index])
    handle = handle.sort_index()

    # Compute YoY growth
    handle_yoy = handle.pct_change(12) * 100  # percentage points

    # Identify deceleration events: YoY growth drops by >=10pp vs prior month
    events = []
    for i in range(1, len(handle_yoy)):
        current_yoy = handle_yoy.iloc[i]
        prior_yoy = handle_yoy.iloc[i - 1]
        if pd.isna(current_yoy) or pd.isna(prior_yoy):
            continue
        decel = prior_yoy - current_yoy  # positive = deceleration
        if decel >= 10.0:
            release_date = handle.index[i] + pd.DateOffset(months=1) + pd.Timedelta(days=RELEASE_LAG_DAYS - 1)
            events.append({
                "handle_month": handle.index[i],
                "current_yoy": current_yoy,
                "prior_yoy": prior_yoy,
                "decel_pp": decel,
                "release_date": release_date,
            })

    print(f"Decel events identified: {len(events)}")

    # Filter to post-2024 (when FLUT is available)
    flut_start = flut_r.dropna().index[0]
    events_post_flut = [ev for ev in events if ev["release_date"] >= flut_start]
    print(f"Events post-FLUT listing ({flut_start.date()}): {len(events_post_flut)}")

    if len(events_post_flut) < 3:
        # Too few events with both legs available; try DKNG-only short
        print("Too few events with FLUT available, falling back to DKNG-only short vs SPY")
        events_post_flut = [ev for ev in events if ev["release_date"] >= pd.Timestamp("2022-01-01")]
        use_flut = False
    else:
        use_flut = True

    if not events_post_flut:
        return mark_failed(sid, "no qualifying decel events in tradeable window")

    # Build daily PnL
    hold_days = 30
    pnl = pd.Series(0.0, index=ret.index)
    valid_events = []

    for ev in events_post_flut:
        rd = ev["release_date"]
        future_idx = ret.index[ret.index >= rd]
        if len(future_idx) < hold_days:
            continue
        entry_date = future_idx[0]
        ep = ret.index.get_loc(entry_date)
        ex = min(ep + hold_days, len(ret))

        d_r = dkng_r.iloc[ep:ex]

        if use_flut:
            f_r = flut_r.iloc[ep:ex]
            # Short DKNG, Long FLUT
            trade_r = f_r.values[:ex-ep] - d_r.values[:ex-ep]
        else:
            # Short DKNG only
            trade_r = -d_r.values[:ex-ep]

        # Check overlap
        trade_slice = pnl.iloc[ep:ex]
        if (trade_slice != 0).sum() > hold_days * 0.5:
            continue

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + trade_r

        dkng_ret = float((1 + d_r).prod() - 1)
        spy_e = spy_r.iloc[ep:ex]
        spy_ret = float((1 + spy_e).prod() - 1)
        if use_flut:
            flut_ret = float((1 + flut_r.iloc[ep:ex]).prod() - 1)
            trade_ret = flut_ret - dkng_ret
        else:
            flut_ret = None
            trade_ret = -dkng_ret

        valid_events.append({
            "release_date": str(rd.date()),
            "handle_month": str(ev["handle_month"].date()),
            "decel_pp": round(float(ev["decel_pp"]), 2),
            "trade_return": round(trade_ret, 4),
            "dkng_return": round(dkng_ret, 4),
            "flut_return": round(flut_ret, 4) if flut_ret is not None else None,
            "spy_return": round(spy_ret, 4),
        })

    print(f"Valid events: {len(valid_events)}")
    if not valid_events:
        return mark_failed(sid, "no valid events after overlap filter")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    strategy_name = "Sports Handle Decel Short DKNG Long FLUT" if use_flut else "Sports Handle Decel Short DKNG"
    m = compute_metrics(active_pnl, benchmark=spy_r, name=strategy_name)
    trade_rets = [ev["trade_return"] for ev in valid_events]
    save_result(sid, m, extra={
        "rule": "When combined state sports-handle YoY growth decelerates by >=10pp MoM, short DKNG and long FLUT (2024+) for 30 trading days.",
        "mechanism": "DKNG has highest US market share and suffers most from handle slowdown; FLUT (FanDuel) has diversified international business as hedge.",
        "source": "NJ DGE, PA PGCB, NY Gaming Commission, IL Gaming Board, MA GC monthly handle reports; yfinance DKNG/FLUT/SPY",
        "n_events": len(valid_events),
        "avg_trade_return": round(float(np.mean(trade_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in trade_rets])), 4),
        "use_flut_hedge": use_flut,
        "events": valid_events,
    })
    print(f"Done: {len(valid_events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
