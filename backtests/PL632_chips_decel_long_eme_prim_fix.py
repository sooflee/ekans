"""PL632_chips_decel_long_eme_prim_fix
Manufacturing Put-in-Place YoY Deceleration -> Long EME/PRIM/FIX (Backlog Conversion).

ENTER long 1/3 EME + 1/3 PRIM + 1/3 FIX when:
  (a) TLMFGCONS YoY < 0 latest, AND
  (b) crossed from positive to negative within last 3 monthly prints, AND
  (c) absolute level in top 10% of trailing 60-month distribution.

EXIT on earliest of:
  (i)   80 trading days,
  (ii)  basket > +0.12,
  (iii) YoY rebounds > +0.05,
  (iv)  basket < -0.08.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL632_chips_decel_long_eme_prim_fix"
    try:
        px = load_prices(["EME", "PRIM", "FIX", "BLD", "SPY"], start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("EME", "PRIM", "FIX", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["TLMFGCONS"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    eme = px["EME"].dropna()
    prim = px["PRIM"].dropna()
    fix = px["FIX"].dropna()
    spy = px["SPY"].dropna()

    parts = pd.concat({"EME": eme, "PRIM": prim, "FIX": fix}, axis=1).dropna()
    if parts.empty or len(parts) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(parts)}")
    norm = parts / parts.iloc[0] * 100
    basket = norm.mean(axis=1)
    basket_r = basket.pct_change()

    tl = fred["TLMFGCONS"].dropna()
    tl_yoy = (tl / tl.shift(12) - 1).dropna()  # monthly
    # Rolling 60-month percentile rank of level
    tl_level_pct = tl.rolling(60).rank(pct=True)
    # Crossed from > 0 to < 0 within last 3 prints
    crossed = pd.Series(False, index=tl_yoy.index)
    yoy_arr = tl_yoy.values
    for i in range(3, len(yoy_arr)):
        if yoy_arr[i] < 0 and any(yoy_arr[i - k] > 0 for k in range(1, 4)):
            crossed.iloc[i] = True

    enter_m = crossed & (tl_yoy < 0) & (tl_level_pct >= 0.90)
    exit_regime_m = tl_yoy > 0.05

    idx = basket.index
    enter_d = enter_m.reindex(idx, method="ffill").fillna(False).astype(bool)
    exit_regime_d = exit_regime_m.reindex(idx, method="ffill").fillna(False).astype(bool)

    df = pd.concat({
        "basket": basket, "basket_r": basket_r,
        "SPY_r": spy.pct_change(),
        "enter_d": enter_d, "exit_regime_d": exit_regime_d,
    }, axis=1).dropna(subset=["basket"])

    trigger_dates = list(df.index[df["enter_d"].values])
    if not trigger_dates:
        return mark_failed(sid, "no signal firings")

    legs = []; events = []
    dates = list(df.index); pos = {d: i for i, d in enumerate(dates)}
    open_until = None
    for t in trigger_dates:
        if open_until is not None and t <= open_until: continue
        i = pos[t]
        if i + 1 >= len(dates): continue
        entry_i = i + 1
        entry_px = df["basket"].iloc[entry_i]
        max_hold = 80
        exit_i = None; exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["basket"].iloc[j]
            ret_so_far = cur_px / entry_px - 1
            if ret_so_far > 0.12:
                exit_i = j; exit_reason = "profit_target"; break
            if ret_so_far < -0.08:
                exit_i = j; exit_reason = "stop"; break
            if df["exit_regime_d"].iloc[j]:
                exit_i = j; exit_reason = "regime_exit"; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i: continue
        leg = df["basket_r"].iloc[entry_i:exit_i + 1]
        if leg.empty: continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(dates[entry_i].date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "ret": round(float((1 + leg).prod() - 1), 4),
        })
        open_until = dates[exit_i]

    if not legs:
        return mark_failed(sid, "no valid events")
    pnl = pd.Series(0.0, index=df.index)
    for leg in legs:
        pnl.loc[leg.index] = pnl.loc[leg.index] + leg.values
    pnl = pnl.loc[legs[0].index[0]:]
    if len(pnl) < 60:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    bench = df["SPY_r"].reindex(pnl.index).fillna(0)
    m = compute_metrics(pnl, benchmark=bench,
                        name="Mfg Construction YoY Decel + High Level -> Long EME/PRIM/FIX")
    save_result(sid, m, extra={
        "rule": ("Long 1/3 EME + 1/3 PRIM + 1/3 FIX when TLMFGCONS YoY just crossed below 0 "
                 "(last 3 prints) AND level in top 10% trailing 60m. Hold up to 80d."),
        "mechanism": ("After mega-cap manufacturing capex boom, YoY decelerates to negative even "
                      "as installed-base still backlog-laden — specialty contractors capture the "
                      "burn-down of backlog."),
        "source": ("FRED TLMFGCONS. yfinance EME, PRIM, FIX, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "2-6 months",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
