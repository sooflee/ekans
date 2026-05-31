"""PL600_vix_basis_compression_short_spy - VIX M1-M2 Basis Compression at Multi-Year Low -> Short SPY / Long VXX

Rule:
  basis = (^VIX3M / ^VIX) - 1.  Compute rolling 5th-percentile and median over trailing 1008
  trading days (~4Y).  ENTER short SPY when:
    (a) basis <= trailing 5th-percentile, AND
    (b) ^VIX close < 13.
  EXIT on earlier of:
    (i)  15 trading days elapsed,
    (ii) ^VIX rises > +30% above entry-day value,
    (iii) basis returns above trailing median.

Daily PnL on SPY-short leg.  Benchmark vs SPY buy-and-hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL600_vix_basis_compression_short_spy"
    try:
        px = load_prices(["SPY", "VXX", "^VIX", "^VIX3M"], start="2007-12-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("SPY", "^VIX", "^VIX3M"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    spy = px["SPY"].dropna()
    vix = px["^VIX"].dropna()
    vix3m = px["^VIX3M"].dropna()

    df = pd.concat({"SPY": spy, "VIX": vix, "VIX3M": vix3m}, axis=1).dropna()
    if df.empty or len(df) < 1100:
        return mark_failed(sid, f"insufficient history: {len(df)}")

    df["basis"] = (df["VIX3M"] / df["VIX"]) - 1.0
    window = 1008
    df["p05"] = df["basis"].rolling(window).quantile(0.05)
    df["pmed"] = df["basis"].rolling(window).median()
    df = df.dropna(subset=["p05", "pmed"])
    if df.empty:
        return mark_failed(sid, "rolling window produced no data")

    df["spy_r"] = df["SPY"].pct_change()
    spy_r_full = df["spy_r"].dropna()

    # Build triggers: basis compressed AND VIX < 13
    cond = (df["basis"] <= df["p05"]) & (df["VIX"] < 13)
    trigger_idx = df.index[cond.fillna(False).values]

    if len(trigger_idx) == 0:
        return mark_failed(sid, "no signal firings")

    # Walk through triggers, but avoid overlap: skip any trigger that
    # falls during an already-open position.
    events = []
    open_until = None
    legs = []

    dates = list(df.index)
    pos_lookup = {d: i for i, d in enumerate(dates)}

    for t in trigger_idx:
        if open_until is not None and t <= open_until:
            continue
        i = pos_lookup[t]
        # Entry effective next bar (no look-ahead)
        if i + 1 >= len(dates):
            continue
        entry_i = i + 1
        entry_date = dates[entry_i]
        entry_vix = df["VIX"].iloc[entry_i]
        entry_median = df["pmed"].iloc[i]  # use signal-day median as exit ref

        max_hold = 15
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            v_now = df["VIX"].iloc[j]
            b_now = df["basis"].iloc[j]
            if v_now > 1.30 * entry_vix:
                exit_i = j
                exit_reason = "vix_spike"
                break
            if b_now > entry_median:
                exit_i = j
                exit_reason = "basis_normalize"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)

        # PnL: short SPY -> daily return = -spy_r over [entry_i, exit_i]
        if exit_i < entry_i:
            continue
        leg = -1.0 * df["spy_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        cum = float((1 + leg).prod() - 1)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "entry_vix": round(float(entry_vix), 2),
            "ret": round(cum, 4),
        })
        open_until = dates[exit_i]

    if not legs:
        return mark_failed(sid, "no valid events after filtering")

    # Build a continuous daily PnL aligned to SPY trading days.
    # Days outside an open position contribute 0.
    pnl = pd.Series(0.0, index=df.index)
    for leg in legs:
        pnl.loc[leg.index] = pnl.loc[leg.index] + leg.values
    # Trim to the first event onward for a fair Sharpe window
    first_dt = legs[0].index[0]
    pnl = pnl.loc[first_dt:]

    if len(pnl) < 60:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    bench = spy_r_full.reindex(pnl.index).fillna(0)
    m = compute_metrics(pnl, benchmark=bench,
                        name="VIX Basis Compression -> Short SPY")
    save_result(sid, m, extra={
        "rule": ("Enter short SPY when (VIX3M/VIX-1) <= trailing 4Y 5th-pctile "
                 "AND VIX < 13. Exit on min(15d, VIX +30% from entry, basis "
                 "above trailing 4Y median)."),
        "mechanism": ("Compressed term structure + low spot vol = dealer short-gamma "
                      "extreme; reversion shock tends to drag SPY when basis re-steepens."),
        "source": "VIX/VIX3M term-structure literature (Donninger 2014; Cheng 2019 VIX risk premium).",
        "n_events": len(events),
        "events_sample": events[:15],
        "ticker_universe": "SPY (short), VIX/VIX3M (signal only)",
        "horizon": "5-15 trading days",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
