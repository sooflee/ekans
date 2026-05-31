"""PL610_china_smartphone_relative_short_aapl
Chinese Domestic Smartphone-Maker Relative Strength -> Short AAPL.

Use Xiaomi (1810.HK) outperformance vs AAPL over 60 trading days as a market-implied
proxy for China premium-smartphone share gain.

ENTER short AAPL when:
  (a) 60d_rel = (1810.HK 60d ret) - (AAPL 60d ret) > 0.25, AND
  (b) AAPL close / AAPL 252d high >= 0.95 (rich entry).

EXIT on earliest of:
  (i)   50 trading days,
  (ii)  20d_rel < -0.05 (reversal),
  (iii) AAPL < -8% below entry close,
  (iv)  AAPL earnings proxy date reached (Jan 30, May 1, Jul 31, Oct 30).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


def _is_earnings_date(d):
    # AAPL fiscal-quarter earnings proxy dates
    m, day = d.month, d.day
    return ((m == 1 and day >= 30)
            or (m == 5 and day <= 2)
            or (m == 7 and day >= 30)
            or (m == 10 and day >= 28))


def main():
    sid = "PL610_china_smartphone_relative_short_aapl"
    try:
        px = load_prices(["AAPL", "1810.HK", "SPY"], start="2018-07-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("AAPL", "1810.HK", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    aapl = px["AAPL"].dropna()
    xm = px["1810.HK"].dropna()
    spy = px["SPY"].dropna()

    # Align on common trading days
    df = pd.concat({"AAPL": aapl, "XM": xm, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["AAPL_r"] = df["AAPL"].pct_change()
    df["XM_r"] = df["XM"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    aapl_60 = df["AAPL"].pct_change(60)
    xm_60 = df["XM"].pct_change(60)
    aapl_20 = df["AAPL"].pct_change(20)
    xm_20 = df["XM"].pct_change(20)
    rel_60 = xm_60 - aapl_60
    rel_20 = xm_20 - aapl_20

    aapl_252_high = df["AAPL"].rolling(252).max()
    near_high = df["AAPL"] / aapl_252_high >= 0.95

    df["rel_60"] = rel_60
    df["rel_20"] = rel_20
    df["near_high"] = near_high

    cond = (df["rel_60"] > 0.25) & (df["near_high"] == True)
    trigger_dates = list(df.index[cond.fillna(False).values])
    if not trigger_dates:
        return mark_failed(sid, "no signal firings")

    legs = []
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
        entry_px = df["AAPL"].iloc[entry_i]
        max_hold = 50
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["AAPL"].iloc[j]
            cur_rel20 = df["rel_20"].iloc[j]
            if cur_px / entry_px - 1 < -0.08:
                exit_i = j
                exit_reason = "profit_target"
                break
            if pd.notna(cur_rel20) and cur_rel20 < -0.05:
                exit_i = j
                exit_reason = "rel_reversal"
                break
            if _is_earnings_date(dates[j]):
                exit_i = j
                exit_reason = "earnings"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = -1.0 * df["AAPL_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "rel_60_at_entry": round(float(df["rel_60"].iloc[i]), 3),
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
                        name="China Smartphone Relative -> Short AAPL")
    save_result(sid, m, extra={
        "rule": ("Short AAPL when 60d (Xiaomi 1810.HK return - AAPL return) > +25% AND "
                 "AAPL within 5% of trailing 252d high. Hold up to 50d."),
        "mechanism": ("China premium-smartphone share gain — proxied by Xiaomi relative "
                      "outperformance — is a leading indicator of AAPL China revenue softening."),
        "source": ("yfinance AAPL, 1810.HK Xiaomi HK, SPY. Market-implied proxy for "
                   "MIIT/CAICT premium-share reports (not in machine-readable form)."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "6-10 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
