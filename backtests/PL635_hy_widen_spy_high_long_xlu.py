"""PL635_hy_widen_spy_high_long_xlu
HY OAS Widening at SPY High -> Long XLU / Short SPY (Defensive Rotation).

ENTER pair (long 1 XLU, short 1 SPY) when:
  (a) BAMLH0A0HYM2 20-trading-day change >= +0.50, AND
  (b) SPY close / SPY trailing 252-day high >= 0.98.

EXIT on earlier of:
  (i)   45 trading days,
  (ii)  pair PnL > +0.05 (profit target),
  (iii) BAMLH0A0HYM2 20-day change <= 0 (regime exit),
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
    sid = "PL635_hy_widen_spy_high_long_xlu"
    try:
        px = load_prices(["XLU", "SPY", "TLT"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("XLU", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["BAMLH0A0HYM2"], start="1996-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    xlu = px["XLU"].dropna()
    spy = px["SPY"].dropna()
    df = pd.concat({"XLU": xlu, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["XLU_r"] = df["XLU"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    hy = fred["BAMLH0A0HYM2"].dropna()
    hy_d = hy.reindex(df.index, method="ffill")
    hy_chg20 = hy_d - hy_d.shift(20)

    spy_252_high = df["SPY"].rolling(252).max()
    near_high = df["SPY"] / spy_252_high >= 0.98

    df["hy"] = hy_d
    df["hy_chg20"] = hy_chg20
    df["near_high"] = near_high.astype(bool)

    cond = (df["hy_chg20"] >= 0.50) & (df["near_high"])
    trigger_dates = list(df.index[cond.fillna(False).values])
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
        max_hold = 45
        cum = 1.0
        exit_i = None; exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            day_r = df["XLU_r"].iloc[j] - df["SPY_r"].iloc[j]
            cum *= (1 + day_r)
            if cum - 1 > 0.05:
                exit_i = j; exit_reason = "profit_target"; break
            if cum - 1 < -0.03:
                exit_i = j; exit_reason = "stop"; break
            cur_chg = df["hy_chg20"].iloc[j]
            if pd.notna(cur_chg) and cur_chg <= 0:
                exit_i = j; exit_reason = "regime_exit"; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i: continue
        leg = (df["XLU_r"].iloc[entry_i:exit_i + 1]
               - df["SPY_r"].iloc[entry_i:exit_i + 1])
        if leg.empty: continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(dates[entry_i].date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "hy_chg20_at_entry": round(float(df["hy_chg20"].iloc[i]), 2),
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
                        name="HY Widening at SPY High -> Long XLU / Short SPY")
    save_result(sid, m, extra={
        "rule": ("Long XLU / short SPY when BAMLH0A0HYM2 20d change >= +50bp AND SPY within 2% "
                 "of 252d high. Hold up to 45d; +5% target / -3% stop / regime exit when HY 20d "
                 "change <= 0."),
        "mechanism": ("Credit-equity divergence: HY widening at SPY high is a classic late-cycle "
                      "risk-off signal — defensive utilities outperform broad market."),
        "source": ("FRED BAMLH0A0HYM2. yfinance XLU, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "1-3 months",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
