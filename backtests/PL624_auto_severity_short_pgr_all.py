"""PL624_auto_severity_short_pgr_all
Auto-Severity Proxy (Used-Car CPI Spike) -> Short PGR/ALL Basket.

ENTER short equal-weight basket (0.5 PGR + 0.5 ALL) when:
  (a) CUSR0000SETA02 (Used Cars CPI) YoY > +0.05, AND
  (b) basket 90d return > +0.15.

EXIT on earlier of:
  (i)   60 trading days,
  (ii)  basket return < -0.08 (profit target),
  (iii) CUSR0000SETA02 YoY < 0 (regime exit),
  (iv)  basket return > +0.06 (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL624_auto_severity_short_pgr_all"
    try:
        px = load_prices(["PGR", "ALL", "TRV", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("PGR", "ALL", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["CUSR0000SETA02"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    pgr = px["PGR"].dropna()
    all_ = px["ALL"].dropna()
    spy = px["SPY"].dropna()
    parts = pd.concat({"PGR": pgr, "ALL": all_}, axis=1).dropna()
    if parts.empty or len(parts) < 400:
        return mark_failed(sid, f"insufficient overlap: {len(parts)}")

    # Normalize each to 100 at common day, equal-weight basket
    norm = parts / parts.iloc[0] * 100
    basket = norm.mean(axis=1)
    basket_r = basket.pct_change()

    cpi = fred["CUSR0000SETA02"].dropna()
    cpi_yoy = (cpi / cpi.shift(12) - 1).dropna()

    idx = basket.index
    cpi_yoy_d = cpi_yoy.reindex(idx, method="ffill")
    basket_90 = basket.pct_change(90)

    df = pd.concat({
        "basket": basket,
        "basket_r": basket_r,
        "SPY_r": spy.pct_change(),
        "cpi_yoy": cpi_yoy_d,
        "basket_90": basket_90,
    }, axis=1).dropna(subset=["basket", "cpi_yoy", "basket_90"])

    cond = (df["cpi_yoy"] > 0.05) & (df["basket_90"] > 0.15)
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
        entry_px = df["basket"].iloc[entry_i]
        max_hold = 60
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["basket"].iloc[j]
            ret_so_far = cur_px / entry_px - 1
            if ret_so_far < -0.08:
                exit_i = j
                exit_reason = "profit_target"
                break
            if ret_so_far > 0.06:
                exit_i = j
                exit_reason = "stop"
                break
            cur_yoy = df["cpi_yoy"].iloc[j]
            if pd.notna(cur_yoy) and cur_yoy < 0:
                exit_i = j
                exit_reason = "regime_exit"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = -1.0 * df["basket_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "cpi_yoy_at_entry": round(float(df["cpi_yoy"].iloc[i]), 3),
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
                        name="Used-Car CPI Spike -> Short PGR/ALL")
    save_result(sid, m, extra={
        "rule": ("Short 0.5 PGR + 0.5 ALL when used-car CPI YoY > +5% AND basket 90d ret > +15%. "
                 "Hold up to 60d; -8% target / +6% stop / regime exit when YoY < 0."),
        "mechanism": ("Used-car CPI inflation proxies total-loss severity for personal-auto "
                      "insurers; when the basket has rallied into a severity regime the combined "
                      "ratio shock isn't yet priced."),
        "source": ("FRED CUSR0000SETA02 (Used Cars CPI). yfinance PGR, ALL, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "4-12 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
