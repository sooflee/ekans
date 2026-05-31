"""PL628_salmon_mortality_long_mowi
Salmon Spot-Price Regime Proxy -> Long MOWI.OL.

ENTER long MOWI.OL when:
  (a) sector basket (MOWI+BAKKA+SALM equal-weight, normalized) / 200d SMA <= 0.90, AND
  (b) BAKKA.OL 20d return >= +0.05.

EXIT on earliest of:
  (i)   60 trading days,
  (ii)  MOWI.OL > +12% above entry,
  (iii) MOWI.OL < -8% below entry,
  (iv)  sector basket 20d return turns negative.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL628_salmon_mortality_long_mowi"
    try:
        px = load_prices(["MOWI.OL", "BAKKA.OL", "SALM.OL", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("MOWI.OL", "BAKKA.OL", "SALM.OL", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    mowi = px["MOWI.OL"].dropna()
    bakka = px["BAKKA.OL"].dropna()
    salm = px["SALM.OL"].dropna()
    spy = px["SPY"].dropna()

    parts = pd.concat({"M": mowi, "B": bakka, "S": salm}, axis=1).dropna()
    if parts.empty or len(parts) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(parts)}")
    norm = parts / parts.iloc[0] * 100
    sector = norm.mean(axis=1)
    sector_sma200 = sector.rolling(200).mean()
    sector_ratio = sector / sector_sma200
    sector_r20 = sector.pct_change(20)

    df = pd.concat({
        "MOWI": mowi.reindex(sector.index),
        "BAKKA": bakka.reindex(sector.index),
        "SALM": salm.reindex(sector.index),
        "SPY_r": spy.pct_change().reindex(sector.index),
        "sector_ratio": sector_ratio,
        "sector_r20": sector_r20,
    }, axis=1).dropna(subset=["MOWI", "BAKKA", "SALM", "sector_ratio"])

    df["MOWI_r"] = df["MOWI"].pct_change()
    df["BAKKA_r20"] = df["BAKKA"].pct_change(20)

    cond = (df["sector_ratio"] <= 0.90) & (df["BAKKA_r20"] >= 0.05)
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
        entry_px = df["MOWI"].iloc[entry_i]
        max_hold = 60
        exit_i = None; exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["MOWI"].iloc[j]
            ret_so_far = cur_px / entry_px - 1
            if ret_so_far > 0.12:
                exit_i = j; exit_reason = "profit_target"; break
            if ret_so_far < -0.08:
                exit_i = j; exit_reason = "stop"; break
            cur_sec20 = df["sector_r20"].iloc[j]
            if pd.notna(cur_sec20) and cur_sec20 < 0:
                exit_i = j; exit_reason = "regime_exit"; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i: continue
        leg = df["MOWI_r"].iloc[entry_i:exit_i + 1]
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
                        name="Salmon Sector Stress -> Long MOWI.OL")
    save_result(sid, m, extra={
        "rule": ("Long MOWI.OL when sector basket / 200d SMA <= 0.90 AND BAKKA.OL 20d ret >= +5%. "
                 "Hold up to 60d; +12% target / -8% stop / regime exit when sector 20d ret < 0."),
        "mechanism": ("Norwegian salmon-farmer sector underperformance proxies elevated mortality/cost "
                      "regime; BAKKA bounce signals regime change and biomass-shortfall premium phase."),
        "source": ("yfinance MOWI.OL, BAKKA.OL, SALM.OL, SPY. Sector-relative price proxy for "
                   "BarentsWatch lice/mortality (not in FRED)."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "4-12 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
