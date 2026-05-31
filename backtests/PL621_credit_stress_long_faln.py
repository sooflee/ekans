"""PL621_credit_stress_long_faln
Credit Stress Cluster (BB-IG OAS Spread Wide) -> Long FALN vs HYG.

ENTER long FALN / short HYG pair when:
  (a) BB-IG spread (BAMLH0A1HYBB - BAMLC0A0CM) > trailing 252d 90th percentile, AND
  (b) FALN 60d return - HYG 60d return <= -0.03 (FALN underperformed).

EXIT on earlier of:
  (i)   80 trading days,
  (ii)  pair PnL > +0.04 (profit target),
  (iii) BB-IG spread tightens below 50th percentile (regime exit),
  (iv)  pair PnL < -0.03 (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL621_credit_stress_long_faln"
    try:
        px = load_prices(["FALN", "HYG", "LQD", "SPY"], start="2016-06-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("FALN", "HYG", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["BAMLH0A1HYBB", "BAMLC0A0CM"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    faln = px["FALN"].dropna()
    hyg = px["HYG"].dropna()
    spy = px["SPY"].dropna()

    df = pd.concat({"FALN": faln, "HYG": hyg, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["FALN_r"] = df["FALN"].pct_change()
    df["HYG_r"] = df["HYG"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    bb = fred["BAMLH0A1HYBB"].dropna()
    ig = fred["BAMLC0A0CM"].dropna()
    spread = (bb - ig).dropna()
    spread_d = spread.reindex(df.index, method="ffill")
    spread_pct = spread_d.rolling(252).rank(pct=True)

    faln_60 = df["FALN"].pct_change(60)
    hyg_60 = df["HYG"].pct_change(60)
    rel_60 = faln_60 - hyg_60

    df["spread"] = spread_d
    df["spread_pct"] = spread_pct
    df["rel_60"] = rel_60

    cond = (df["spread_pct"] > 0.90) & (df["rel_60"] <= -0.03)
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
        max_hold = 80
        cum = 1.0
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            day_r = df["FALN_r"].iloc[j] - df["HYG_r"].iloc[j]
            cum *= (1 + day_r)
            if cum - 1 > 0.04:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cum - 1 < -0.03:
                exit_i = j
                exit_reason = "stop"
                break
            cur_pct = df["spread_pct"].iloc[j]
            if pd.notna(cur_pct) and cur_pct < 0.50:
                exit_i = j
                exit_reason = "regime_exit"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = (df["FALN_r"].iloc[entry_i:exit_i + 1]
               - df["HYG_r"].iloc[entry_i:exit_i + 1])
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "spread_pct_at_entry": round(float(df["spread_pct"].iloc[i]), 3),
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
                        name="Credit Stress (BB-IG wide) -> Long FALN / Short HYG")
    save_result(sid, m, extra={
        "rule": ("Long FALN / short HYG pair when BB-IG OAS spread > 90th pctile (trailing 252d) "
                 "AND FALN 60d ret - HYG 60d ret <= -3%. Hold up to 80d; +4% target / -3% stop / "
                 "regime exit when spread tightens below 50th pctile."),
        "mechanism": ("Forced-seller overshoot in fallen-angel ETF (FALN) during credit stress "
                      "tends to mean-revert when stress eases — pair captures the FALN-specific "
                      "recovery."),
        "source": ("FRED BAMLH0A1HYBB, BAMLC0A0CM. yfinance FALN, HYG."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "3-6 months",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
