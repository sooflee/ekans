"""PL617_pharma_ppi_negative_short_large_biotech
Pharma PPI YoY Negative + Biotech Rich -> Short GILD/VRTX/REGN Basket.

ENTER short equal-weight basket of GILD/VRTX/REGN when:
  (a) FRED WPU0638 (PPI Pharma Prep) YoY < 0 for 2 consecutive monthly prints, AND
  (b) basket close / basket 200d SMA >= 1.15 on entry day (price extended).

EXIT on earlier of:
  (i)   80 trading days,
  (ii)  basket > +10% above entry (stop),
  (iii) basket < -12% below entry (profit target),
  (iv)  WPU0638 YoY flips positive for 2 consecutive months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL617_pharma_ppi_negative_short_large_biotech"
    try:
        px = load_prices(["GILD", "VRTX", "REGN", "XLV", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    for c in ("GILD", "VRTX", "REGN", "SPY"):
        if c not in px.columns:
            return mark_failed(sid, f"missing column {c}")

    try:
        fred = load_fred(["WPU0638"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    gild = px["GILD"].dropna()
    vrtx = px["VRTX"].dropna()
    regn = px["REGN"].dropna()
    spy = px["SPY"].dropna()

    # Build equal-weight basket of (normalized) prices
    parts = pd.concat({"GILD": gild, "VRTX": vrtx, "REGN": regn}, axis=1).dropna()
    if parts.empty or len(parts) < 400:
        return mark_failed(sid, f"insufficient basket overlap: {len(parts)}")

    # Normalize each to 100 at the first common day
    norm = parts / parts.iloc[0] * 100
    basket = norm.mean(axis=1)  # equal-weight composite
    basket_r = basket.pct_change()
    basket_sma200 = basket.rolling(200).mean()
    basket_ratio = basket / basket_sma200

    ppi = fred["WPU0638"].dropna()
    ppi_yoy = (ppi / ppi.shift(12) - 1) * 100  # monthly %

    # Joint: latest 2 monthly prints < 0
    ppi_neg2 = (ppi_yoy < 0) & (ppi_yoy.shift(1) < 0)
    ppi_pos2 = (ppi_yoy > 0) & (ppi_yoy.shift(1) > 0)

    # Forward-fill monthly to daily on basket index
    idx = basket.index
    ppi_neg2_d = ppi_neg2.reindex(idx, method="ffill").fillna(False)
    ppi_pos2_d = ppi_pos2.reindex(idx, method="ffill").fillna(False)

    df = pd.concat({
        "basket": basket,
        "basket_r": basket_r,
        "ratio": basket_ratio,
        "SPY_r": spy.pct_change(),
        "ppi_neg2": ppi_neg2_d.astype(bool),
        "ppi_pos2": ppi_pos2_d.astype(bool),
    }, axis=1).dropna(subset=["basket", "ratio"])

    cond = (df["ppi_neg2"]) & (df["ratio"] >= 1.15)
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
        max_hold = 80
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["basket"].iloc[j]
            cur_pos2 = df["ppi_pos2"].iloc[j]
            ret_so_far = cur_px / entry_px - 1
            if ret_so_far > 0.10:
                exit_i = j
                exit_reason = "stop"
                break
            if ret_so_far < -0.12:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cur_pos2:
                exit_i = j
                exit_reason = "ppi_flip"
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
            "ratio_at_entry": round(float(df["ratio"].iloc[entry_i]), 3),
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
                        name="Pharma PPI Negative + Biotech Rich -> Short GILD/VRTX/REGN")
    save_result(sid, m, extra={
        "rule": ("Short equal-weight GILD/VRTX/REGN when WPU0638 (PPI Pharma) YoY < 0 "
                 "for 2 prints AND basket / 200d SMA >= 1.15. Hold up to 80d; +10% stop / "
                 "-12% target / PPI flip exit."),
        "mechanism": ("Pharma producer-price deflation combined with rich biotech multiples "
                      "signals margin compression at peak optimism — large biotech "
                      "downside-skewed."),
        "source": ("FRED WPU0638 (PPI Pharma Prep). yfinance GILD, VRTX, REGN, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "2-6 months",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
