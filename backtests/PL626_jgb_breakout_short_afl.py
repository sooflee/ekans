"""PL626_jgb_breakout_short_afl
JGB 10Y Breakout + Yen Strength -> Short AFL/MET Basket.

ENTER short 0.5 AFL + 0.5 MET when:
  (a) IRLTLT01JPM156N latest monthly > 1.50, AND
  (b) IRLTLT01JPM156N 2-month change >= +0.30, AND
  (c) JPY=X 20-day change <= -0.03.

EXIT on earlier of:
  (i)  80 trading days,
  (ii) basket short return < -0.08 (profit target),
  (iii) IRLTLT01JPM156N < 1.30 (regime exit),
  (iv) basket short return > +0.06 (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL626_jgb_breakout_short_afl"
    try:
        px = load_prices(["AFL", "MET", "PRU", "JPY=X", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("AFL", "MET", "JPY=X", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["IRLTLT01JPM156N"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    afl = px["AFL"].dropna()
    met = px["MET"].dropna()
    jpyx = px["JPY=X"].dropna()
    spy = px["SPY"].dropna()
    parts = pd.concat({"AFL": afl, "MET": met}, axis=1).dropna()
    if parts.empty or len(parts) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(parts)}")
    norm = parts / parts.iloc[0] * 100
    basket = norm.mean(axis=1)
    basket_r = basket.pct_change()

    jgb = fred["IRLTLT01JPM156N"].dropna()
    jgb_d_change2m = (jgb - jgb.shift(2)).dropna()

    idx = basket.index
    jgb_d = jgb.reindex(idx, method="ffill")
    jgb_d_chg2m = jgb_d_change2m.reindex(idx, method="ffill")
    jpyx_d = jpyx.reindex(idx).ffill()
    jpyx_chg20 = jpyx_d.pct_change(20)

    df = pd.concat({
        "basket": basket, "basket_r": basket_r,
        "SPY_r": spy.pct_change(),
        "jgb": jgb_d, "jgb_chg2m": jgb_d_chg2m,
        "jpyx_chg20": jpyx_chg20,
    }, axis=1).dropna(subset=["basket", "jgb", "jpyx_chg20"])

    cond = (df["jgb"] > 1.50) & (df["jgb_chg2m"] >= 0.30) & (df["jpyx_chg20"] <= -0.03)
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
        entry_px = df["basket"].iloc[entry_i]
        max_hold = 80
        exit_i = None; exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["basket"].iloc[j]
            ret_so_far = -1 * (cur_px / entry_px - 1)  # short PnL
            if ret_so_far > 0.08:
                exit_i = j; exit_reason = "profit_target"; break
            if ret_so_far < -0.06:
                exit_i = j; exit_reason = "stop"; break
            cur_jgb = df["jgb"].iloc[j]
            if pd.notna(cur_jgb) and cur_jgb < 1.30:
                exit_i = j; exit_reason = "regime_exit"; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i: continue
        leg = -1.0 * df["basket_r"].iloc[entry_i:exit_i + 1]
        if leg.empty: continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(dates[entry_i].date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "jgb_at_entry": round(float(df["jgb"].iloc[entry_i]), 2),
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
                        name="JGB Breakout + Yen Strength -> Short AFL/MET")
    save_result(sid, m, extra={
        "rule": ("Short 0.5 AFL + 0.5 MET when JGB 10y > 1.5% AND 2m rise >= 30bp AND "
                 "JPY=X 20d ret <= -3%. Hold up to 80d; -8% target / +6% stop / regime exit when JGB<1.3%."),
        "mechanism": ("Japan-exposed US life insurers (AFL, MET) face duration + yen-strength "
                      "headwinds when JGB yields break out and yen strengthens — book values "
                      "and earnings translation hit."),
        "source": ("FRED IRLTLT01JPM156N. yfinance AFL, MET, JPY=X, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "6-16 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
