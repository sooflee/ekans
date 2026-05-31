"""PL605_sarb_real_rate_emhy_long_eza
SARB Real Rate >4% + EM HY OAS Compressing -> Long EZA (ZAR Carry Resumption)

ENTER LONG EZA when ALL of:
  (a) SARB real = INTDSRZAM193N - ZAFCPI 12m YoY > 4.0,
  (b) BAMLEMHGHYCRPIUSOAS has tightened by >50 bp over trailing 40 trading days,
  (c) EZA close not more than 1.5 sigma above its 60-day SMA (avoid chasing).

EXIT on earlier of:
  (i)   50 trading days,
  (ii)  SARB real falls below 3.0,
  (iii) EM HY OAS widens >30bp over trailing 20 trading days from current level,
  (iv)  EZA up >+8% above entry close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL605_sarb_real_rate_emhy_long_eza"
    try:
        px = load_prices(["EZA", "USDZAR=X", "SPY"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty or "EZA" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing EZA/SPY")

    try:
        # INTDSRZAM193N (SARB discount rate) FRED series ended in 2013-04 — use
        # IR3TIB01ZAM156N (ZA 3-month interbank rate, monthly), which is a clean
        # proxy for the SARB repo rate and is current through 2026.
        # BAMLEMHGHYCRPIUSOAS (HY EM corporate constrained OAS) is not in FRED;
        # substitute BAMLEMCBPIOAS (ICE BofA EM Corporate Plus Index OAS).
        fred = load_fred(["IR3TIB01ZAM156N", "ZAFCPIALLMINMEI",
                          "BAMLEMCBPIOAS"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")
    if fred is None or fred.empty:
        return mark_failed(sid, "no FRED data")

    eza = px["EZA"].dropna()
    spy = px["SPY"].dropna()

    sarb_disc = fred["IR3TIB01ZAM156N"].dropna()  # monthly, % (3M IBOR ZA proxy)
    za_cpi = fred["ZAFCPIALLMINMEI"].dropna()   # monthly index
    em_oas = fred["BAMLEMCBPIOAS"].dropna()  # daily, % spread (EM Corp Plus OAS proxy)

    za_cpi_yoy = (za_cpi / za_cpi.shift(12) - 1) * 100  # %
    sarb_real_m = (sarb_disc - za_cpi_yoy).dropna()     # monthly, %

    idx = eza.index
    sarb_real_d = sarb_real_m.reindex(idx, method="ffill")
    em_oas_d = em_oas.reindex(idx, method="ffill")

    sma60 = eza.rolling(60).mean()
    eza_ret = eza.pct_change()
    sigma60 = eza_ret.rolling(60).std()

    em_oas_delta_40 = em_oas_d - em_oas_d.shift(40)
    em_oas_delta_20 = em_oas_d - em_oas_d.shift(20)

    df = pd.concat({
        "EZA": eza,
        "EZA_r": eza_ret,
        "SPY_r": spy.pct_change(),
        "sarb_real": sarb_real_d,
        "em_oas": em_oas_d,
        "em_oas_d40": em_oas_delta_40,
        "em_oas_d20": em_oas_delta_20,
        "sma60": sma60,
        "sigma60": sigma60,
    }, axis=1).dropna(subset=["EZA", "sarb_real", "em_oas_d40", "sma60", "sigma60"])

    if df.empty:
        return mark_failed(sid, "empty combined frame")

    # Not-extended: EZA / SMA60 - 1 <= 1.5 * sigma60
    not_extended = (df["EZA"] / df["sma60"] - 1) <= (1.5 * df["sigma60"])
    cond = (
        (df["sarb_real"] > 4.0)
        & (df["em_oas_d40"] <= -0.50)
        & not_extended
    )
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
        entry_px = df["EZA"].iloc[entry_i]
        max_hold = 50

        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, min(entry_i + max_hold, len(dates))):
            cur_px = df["EZA"].iloc[j]
            cur_sarb = df["sarb_real"].iloc[j]
            cur_em_d20 = df["em_oas_d20"].iloc[j]
            if cur_px / entry_px - 1 > 0.08:
                exit_i = j
                exit_reason = "profit_target"
                break
            if cur_sarb < 3.0:
                exit_i = j
                exit_reason = "sarb_real_drop"
                break
            if pd.notna(cur_em_d20) and cur_em_d20 > 0.30:
                exit_i = j
                exit_reason = "oas_widen"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = df["EZA_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_date": str(dates[i].date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "entry_sarb_real": round(float(df["sarb_real"].iloc[entry_i]), 2),
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
                        name="SARB Real-Rate + EM HY Compression -> Long EZA")

    save_result(sid, m, extra={
        "rule": ("Long EZA when SARB real rate >4%, EM HY OAS tightens >50bp in 40d, "
                 "EZA not extended >1.5σ above 60d SMA. Hold up to 50d."),
        "mechanism": ("ZAR carry resumption: when SARB real rate is rich and EM credit "
                      "premia compress, ZAR + SA equities catch a bid as carry trades reload."),
        "source": ("FRED IR3TIB01ZAM156N (ZA 3M IBOR proxy for SARB repo), "
                   "ZAFCPIALLMINMEI, BAMLEMCBPIOAS (EM Corp Plus OAS proxy "
                   "for EM HY). yfinance EZA."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "4-10 weeks",
        "counter_signal": False,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
