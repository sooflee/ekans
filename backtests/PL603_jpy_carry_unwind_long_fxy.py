"""PL603_jpy_carry_unwind_long_fxy
JPY Real-Rate Gap Compression + USD/JPY Spike -> Long FXY (Carry Unwind)

Build a US-JP 10y real-rate gap from FRED:
    real_gap = DFII10 - (IRLTLT01JPM156N - JPNCPIALLMINMEI 12m YoY)
Track 20-trading-day delta.  Track 20-trading-day USD/JPY (JPY=X) change.

ENTER LONG FXY when:
  (a) delta_real_gap_4w <= -0.25 (US real rate falls vs JP real by >25bp), AND
  (b) delta_JPYX_4w >= +0.03 (JPY weakened >=3% in prior 4w).

EXIT on earlier of:
  (i)  20 trading days,
  (ii) FXY > +5% above entry close,
  (iii) delta_real_gap_4w rises back above 0.

Counter-signal against long-SPY default; primary PnL = long FXY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL603_jpy_carry_unwind_long_fxy"
    try:
        px = load_prices(["FXY", "SPY", "JPY=X"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("FXY", "SPY", "JPY=X"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["DFII10", "IRLTLT01JPM156N", "JPNCPIALLMINMEI"],
                         start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    fxy = px["FXY"].dropna()
    spy = px["SPY"].dropna()
    jpyx = px["JPY=X"].dropna()

    # Build daily real-rate gap, forward-filled to trading days
    f = fred.copy()
    # Compute JP CPI 12m YoY (monthly), then forward fill to daily
    jp_cpi = f["JPNCPIALLMINMEI"].dropna()
    jp_cpi_yoy = (jp_cpi / jp_cpi.shift(12) - 1) * 100  # in percent

    # JP nominal 10y yield (monthly, %)
    jp_y = f["IRLTLT01JPM156N"].dropna()
    # JP real = nominal - inflation YoY  (need same monthly index)
    jp_real_m = (jp_y - jp_cpi_yoy).dropna()  # monthly %

    # US 10y TIPS yield (daily, %)
    us_real = f["DFII10"].dropna()  # daily

    # Forward-fill JP real (monthly) to daily on FXY trading-day index
    idx = fxy.index
    jp_real_d = jp_real_m.reindex(idx, method="ffill")
    us_real_d = us_real.reindex(idx, method="ffill")

    real_gap = (us_real_d - jp_real_d).dropna()
    if len(real_gap) < 200:
        return mark_failed(sid, f"real_gap too short: {len(real_gap)}")

    jpyx_d = jpyx.reindex(idx).ffill()
    fxy_r = fxy.pct_change()
    spy_r = spy.pct_change()

    # 20-trading-day changes
    d_gap_20 = real_gap - real_gap.shift(20)
    d_jpyx_20 = jpyx_d.pct_change(20)

    df = pd.concat({
        "FXY": fxy,
        "FXY_r": fxy_r,
        "SPY_r": spy_r,
        "real_gap": real_gap,
        "d_gap_20": d_gap_20,
        "d_jpyx_20": d_jpyx_20,
    }, axis=1).dropna(subset=["FXY", "d_gap_20", "d_jpyx_20"])

    if df.empty:
        return mark_failed(sid, "empty combined frame")

    cond = (df["d_gap_20"] <= -0.25) & (df["d_jpyx_20"] >= 0.03)
    trigger_dates = list(df.index[cond.fillna(False).values])
    if not trigger_dates:
        return mark_failed(sid, "no signal firings")

    # Walk through, no-overlap
    legs_fxy = []
    legs_spy = []
    events = []
    dates = list(df.index)
    pos = {d: i for i, d in enumerate(dates)}
    open_until = None

    for t in trigger_dates:
        if open_until is not None and t <= open_until:
            continue
        i = pos[t]
        if i + 1 >= len(dates):
            continue
        entry_i = i + 1
        entry_date = dates[entry_i]
        entry_px = df["FXY"].iloc[entry_i]
        max_hold = 20

        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["FXY"].iloc[j]
            cur_gap_d = df["d_gap_20"].iloc[j]
            if cur_px / entry_px - 1 > 0.05:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cur_gap_d > 0:
                exit_i = j
                exit_reason = "gap_revert"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue

        leg_fxy = df["FXY_r"].iloc[entry_i:exit_i + 1]
        leg_spy_short = -1.0 * df["SPY_r"].iloc[entry_i:exit_i + 1]
        if leg_fxy.empty:
            continue

        legs_fxy.append(leg_fxy)
        legs_spy.append(leg_spy_short)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "ret_fxy": round(float((1 + leg_fxy).prod() - 1), 4),
            "ret_spy_short": round(float((1 + leg_spy_short).prod() - 1), 4),
        })
        open_until = dates[exit_i]

    if not legs_fxy:
        return mark_failed(sid, "no valid events")

    pnl = pd.Series(0.0, index=df.index)
    for leg in legs_fxy:
        pnl.loc[leg.index] = pnl.loc[leg.index] + leg.values
    pnl = pnl.loc[legs_fxy[0].index[0]:]

    # Also build secondary SPY-short pnl for comparison
    pnl_spy_short = pd.Series(0.0, index=df.index)
    for leg in legs_spy:
        pnl_spy_short.loc[leg.index] = pnl_spy_short.loc[leg.index] + leg.values
    pnl_spy_short = pnl_spy_short.loc[legs_spy[0].index[0]:]

    if len(pnl) < 60:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    bench = df["SPY_r"].reindex(pnl.index).fillna(0)
    m = compute_metrics(pnl, benchmark=bench,
                        name="JPY Carry Unwind -> Long FXY")

    # Secondary metrics: SPY-short leg
    spy_short_metrics = compute_metrics(
        pnl_spy_short, benchmark=bench,
        name="JPY Carry Unwind -> Short SPY (secondary)")

    save_result(sid, m, extra={
        "rule": ("Long FXY when 20d US-JP real-rate gap compresses by >25bp AND "
                 "USD/JPY rose >+3% in prior 20d. Exit on min(20d, FXY +5%, gap reverts)."),
        "mechanism": ("Carry-trade unwind: crowded short-JPY funding positions reverse "
                      "when US-JP real-rate differential narrows; FXY (yen) rallies."),
        "source": ("FRED DFII10, IRLTLT01JPM156N, JPNCPIALLMINMEI; yfinance FXY/JPY=X. "
                   "Anchor: 2024-08-05 carry-unwind episode."),
        "n_events": len(events),
        "events_sample": events[:15],
        "secondary_spy_short": {
            "sharpe": spy_short_metrics.get("sharpe"),
            "cagr": spy_short_metrics.get("cagr"),
            "n_days": spy_short_metrics.get("n_days"),
        },
        "horizon": "2-6 weeks",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
