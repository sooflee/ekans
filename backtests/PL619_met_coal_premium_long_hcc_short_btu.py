"""PL619_met_coal_premium_long_hcc_short_btu
Met Coal Premium Strength Proxy -> Long HCC / Short BTU Pair.

ENTER pair (long HCC, short BTU notional-matched) when:
  (a) METC 60d return - BTU 60d return >= +0.10, AND
  (b) HCC close <= HCC 50d SMA * 1.15 (not extended).

EXIT on earlier of:
  (i)   60 trading days,
  (ii)  pair return > +0.12 (profit target),
  (iii) METC 20d return - BTU 20d return < 0 (regime reversal),
  (iv)  pair return < -0.08 (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL619_met_coal_premium_long_hcc_short_btu"
    try:
        px = load_prices(["HCC", "BTU", "METC", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("HCC", "BTU", "METC", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    hcc = px["HCC"].dropna()
    btu = px["BTU"].dropna()
    metc = px["METC"].dropna()
    spy = px["SPY"].dropna()

    df = pd.concat({"HCC": hcc, "BTU": btu, "METC": metc, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["HCC_r"] = df["HCC"].pct_change()
    df["BTU_r"] = df["BTU"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    metc_60 = df["METC"].pct_change(60)
    btu_60 = df["BTU"].pct_change(60)
    metc_20 = df["METC"].pct_change(20)
    btu_20 = df["BTU"].pct_change(20)
    rel_60 = metc_60 - btu_60
    rel_20 = metc_20 - btu_20

    hcc_sma50 = df["HCC"].rolling(50).mean()
    not_extended = df["HCC"] <= hcc_sma50 * 1.15

    df["rel_60"] = rel_60
    df["rel_20"] = rel_20
    df["not_ext"] = not_extended

    cond = (df["rel_60"] >= 0.10) & (df["not_ext"] == True)
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
        max_hold = 60
        # cumulative pair return tracker
        cum = 1.0
        exit_i = None
        exit_reason = "time"
        last_j = entry_i
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            day_r = df["HCC_r"].iloc[j] - df["BTU_r"].iloc[j]
            cum *= (1 + day_r)
            if cum - 1 > 0.12:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cum - 1 < -0.08:
                exit_i = j
                exit_reason = "stop"
                break
            cur_rel20 = df["rel_20"].iloc[j]
            if pd.notna(cur_rel20) and cur_rel20 < 0:
                exit_i = j
                exit_reason = "rel_reversal"
                break
            last_j = j
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = (df["HCC_r"].iloc[entry_i:exit_i + 1]
               - df["BTU_r"].iloc[entry_i:exit_i + 1])
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
                        name="Met Coal Premium -> Long HCC / Short BTU")
    save_result(sid, m, extra={
        "rule": ("Long HCC / short BTU pair when METC-BTU 60d rel return >= +10% AND "
                 "HCC <= 1.15 x 50d SMA. Hold up to 60d; +12% target / -8% stop / 20d rel reversal."),
        "mechanism": ("Met-coal premium spread (HCC-PCI) widening is leadingly proxied by "
                      "Ramaco (pure met) outperforming Peabody (diversified); HCC (Warrior, "
                      "pure premium HCC) is the cleanest beneficiary."),
        "source": ("yfinance HCC, BTU, METC, SPY. Equity-pair proxy for paywalled "
                   "Argus/Platts met-coal index."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "6-12 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
