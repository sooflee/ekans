"""PL613_hog_herd_contraction_long_he
Lean Hog Breeding-Herd Contraction Proxy -> Long HE=F Front-Month.

Price-regime proxy for USDA Quarterly Hogs and Pigs breeding-herd contraction:
sustained low-margin lean-hog price regime is the leading indicator that drives
herd liquidation (cobweb model).

ENTER LONG HE=F when:
  (a) HE=F close percentile rank over trailing 252d <= 0.15 (bottom 15th), AND
  (b) HE=F close < trailing 252d SMA for the prior 60 consecutive trading days.

EXIT on earlier of:
  (i)  100 trading days (~5 months — captures biological lag),
  (ii) HE=F up >+18% above entry close (profit target).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL613_hog_herd_contraction_long_he"
    try:
        px = load_prices(["HE=F", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty or "HE=F" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing HE=F/SPY")

    he = px["HE=F"].dropna()
    spy = px["SPY"].dropna()
    df = pd.concat({"HE": he, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["HE_r"] = df["HE"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    # 252-day percentile rank
    df["sma252"] = df["HE"].rolling(252).mean()
    df["pct_rank"] = df["HE"].rolling(252).rank(pct=True)
    # Consecutive days below 252d SMA
    below = (df["HE"] < df["sma252"]).astype(int)
    streak = below * (below.groupby((below != below.shift()).cumsum()).cumcount() + 1)
    df["below_streak"] = streak

    cond = (df["pct_rank"] <= 0.15) & (df["below_streak"] >= 60)
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
        entry_px = df["HE"].iloc[entry_i]
        max_hold = 100
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["HE"].iloc[j]
            if cur_px / entry_px - 1 > 0.18:
                exit_i = j
                exit_reason = "profit_target"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = df["HE_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "entry_pct_rank": round(float(df["pct_rank"].iloc[i]), 3),
            "below_streak_at_trigger": int(df["below_streak"].iloc[i]),
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
                        name="Hog Herd Contraction -> Long HE=F")

    save_result(sid, m, extra={
        "rule": ("Long HE=F when its 252d percentile rank <=0.15 AND it has been below "
                 "its 252d SMA for 60+ consecutive trading days. Hold up to 100d, +18% target."),
        "mechanism": ("Cobweb hog cycle: sustained low-margin price regime drives herd "
                      "liquidation; ~10-month biological lag then squeezes supply and prices rally."),
        "source": ("yfinance HE=F. Price-regime proxy for USDA NASS Quarterly Hogs and Pigs "
                   "(PDF-only, not in FRED)."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "3-6 months",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
