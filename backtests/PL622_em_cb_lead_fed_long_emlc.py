"""PL622_em_cb_lead_fed_long_emlc
EM CB Cutting Lead vs Fed Proxy -> Long EMLC / Short EMB Pair.

ENTER pair (long EMLC / short EMB) when:
  (a) 12m change in BR rate (INTDSRBRM193N) - 12m change in FEDFUNDS <= -1.5, AND
  (b) EMLC 60d return - EMB 60d return >= 0.

EXIT on earlier of:
  (i)   60 trading days,
  (ii)  pair PnL > +0.05 (profit target),
  (iii) delta_BR - delta_Fed > -0.5 (gap closed),
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
    sid = "PL622_em_cb_lead_fed_long_emlc"
    try:
        px = load_prices(["EMLC", "EMB", "LEMB", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("EMLC", "EMB", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["INTDSRBRM193N", "FEDFUNDS"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    emlc = px["EMLC"].dropna()
    emb = px["EMB"].dropna()
    spy = px["SPY"].dropna()
    df = pd.concat({"EMLC": emlc, "EMB": emb, "SPY": spy}, axis=1).dropna()
    if df.empty or len(df) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(df)}")

    df["EMLC_r"] = df["EMLC"].pct_change()
    df["EMB_r"] = df["EMB"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    br = fred["INTDSRBRM193N"].dropna()
    ff = fred["FEDFUNDS"].dropna()
    delta_br_12m = (br - br.shift(12)).dropna()  # monthly delta over 12m
    delta_ff_12m = (ff - ff.shift(12)).dropna()
    gap = (delta_br_12m - delta_ff_12m).dropna()

    gap_d = gap.reindex(df.index, method="ffill")

    emlc_60 = df["EMLC"].pct_change(60)
    emb_60 = df["EMB"].pct_change(60)
    rel_60 = emlc_60 - emb_60

    df["gap"] = gap_d
    df["rel_60"] = rel_60

    cond = (df["gap"] <= -1.5) & (df["rel_60"] >= 0)
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
        cum = 1.0
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            day_r = df["EMLC_r"].iloc[j] - df["EMB_r"].iloc[j]
            cum *= (1 + day_r)
            if cum - 1 > 0.05:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cum - 1 < -0.03:
                exit_i = j
                exit_reason = "stop"
                break
            cur_gap = df["gap"].iloc[j]
            if pd.notna(cur_gap) and cur_gap > -0.5:
                exit_i = j
                exit_reason = "gap_closed"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = (df["EMLC_r"].iloc[entry_i:exit_i + 1]
               - df["EMB_r"].iloc[entry_i:exit_i + 1])
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "gap_at_entry": round(float(df["gap"].iloc[i]), 2),
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
                        name="EM CB Lead vs Fed -> Long EMLC / Short EMB")
    save_result(sid, m, extra={
        "rule": ("Long EMLC / short EMB when 12m BR rate change minus 12m Fed Funds change <= "
                 "-1.5 AND EMLC 60d ret - EMB 60d ret >= 0. Hold up to 60d; +5% target / -3% stop / "
                 "gap-closed exit."),
        "mechanism": ("EM central banks (Brazil as proxy) cutting ahead of the Fed → EM local-rates "
                      "rally faster than EM USD bonds; pair captures the duration-relative-FX "
                      "spread."),
        "source": ("FRED INTDSRBRM193N, FEDFUNDS. yfinance EMLC, EMB."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "2-4 months",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
