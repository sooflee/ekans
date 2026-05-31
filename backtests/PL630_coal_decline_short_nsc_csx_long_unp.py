"""PL630_coal_decline_short_nsc_csx_long_unp
Coal Decline Proxy -> Long UNP / Short NSC+CSX Pair.

ENTER pair (long 1 UNP, short 0.5 NSC + 0.5 CSX) when:
  (a) FRED IPN2121N YoY <= -0.15, AND
  (b) each of last 2 monthly prints shows YoY <= -0.10.

EXIT on earliest of:
  (i)   60 trading days,
  (ii)  pair PnL > +0.06 (profit target),
  (iii) FRED IPN2121N YoY recovers above -0.05 (regime exit),
  (iv)  pair PnL < -0.04 (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL630_coal_decline_short_nsc_csx_long_unp"
    try:
        px = load_prices(["NSC", "CSX", "UNP", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("NSC", "CSX", "UNP", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["IPN2121N"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    nsc = px["NSC"].dropna()
    csx = px["CSX"].dropna()
    unp = px["UNP"].dropna()
    spy = px["SPY"].dropna()

    df = pd.concat({"NSC": nsc, "CSX": csx, "UNP": unp, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["NSC_r"] = df["NSC"].pct_change()
    df["CSX_r"] = df["CSX"].pct_change()
    df["UNP_r"] = df["UNP"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    ip = fred["IPN2121N"].dropna()
    ip_yoy = (ip / ip.shift(12) - 1).dropna()
    # Sustained: last 2 prints both <= -0.10, and latest also <= -0.15
    yoy_below_10_prev = (ip_yoy.shift(1) <= -0.10) & (ip_yoy <= -0.10)
    yoy_below_15 = ip_yoy <= -0.15
    enter_m = yoy_below_10_prev & yoy_below_15

    enter_d = enter_m.reindex(df.index, method="ffill").fillna(False).astype(bool)
    exit_regime_d = (ip_yoy > -0.05).reindex(df.index, method="ffill").fillna(False).astype(bool)
    df["enter_d"] = enter_d
    df["exit_regime_d"] = exit_regime_d

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
        max_hold = 60
        cum = 1.0
        exit_i = None; exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            day_r = (df["UNP_r"].iloc[j]
                     - 0.5 * df["NSC_r"].iloc[j]
                     - 0.5 * df["CSX_r"].iloc[j])
            cum *= (1 + day_r)
            if cum - 1 > 0.06:
                exit_i = j; exit_reason = "profit_target"; break
            if cum - 1 < -0.04:
                exit_i = j; exit_reason = "stop"; break
            if df["exit_regime_d"].iloc[j]:
                exit_i = j; exit_reason = "regime_exit"; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i: continue
        leg = (df["UNP_r"].iloc[entry_i:exit_i + 1]
               - 0.5 * df["NSC_r"].iloc[entry_i:exit_i + 1]
               - 0.5 * df["CSX_r"].iloc[entry_i:exit_i + 1])
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
                        name="Coal Decline -> Long UNP / Short NSC+CSX")
    save_result(sid, m, extra={
        "rule": ("Long UNP / short 0.5 NSC + 0.5 CSX when FRED IPN2121N YoY <= -15% AND "
                 "2 consecutive prints <= -10%. Hold up to 60d; +6% target / -4% stop / regime exit "
                 "when YoY > -5%."),
        "mechanism": ("Coal-mining IP collapse hits Eastern (NSC/CSX) coal-rail revenue more than "
                      "Western (UNP); pair captures the relative-earnings divergence."),
        "source": ("FRED IPN2121N. yfinance NSC, CSX, UNP, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "4-12 weeks",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
