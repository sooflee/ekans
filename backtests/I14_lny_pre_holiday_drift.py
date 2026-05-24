"""
I-14 Lunar New Year pre-holiday drift on EWH (Hong Kong) and EWS (Singapore).

Hard-code Lunar New Year (Chinese New Year) dates 2000-2026. For each year,
hold long EWH and EWS from LNY-15 trading sessions through LNY-1 session.
Annualize captured-return-per-window vs time-in-market.

ETFs: EWH (Hong Kong) iShares MSCI HK; EWS (Singapore) iShares MSCI Singapore.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)

LNY_DATES = {
    2000: "2000-02-05",
    2001: "2001-01-24",
    2002: "2002-02-12",
    2003: "2003-02-01",
    2004: "2004-01-22",
    2005: "2005-02-09",
    2006: "2006-01-29",
    2007: "2007-02-18",
    2008: "2008-02-07",
    2009: "2009-01-26",
    2010: "2010-02-14",
    2011: "2011-02-03",
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
    2026: "2026-02-17",
}


def main():
    try:
        px = load_prices(["EWH", "EWS"], start="1999-06-01")
    except Exception as e:
        return mark_failed("I14_lny_pre_holiday_drift", f"Price load failed: {e}")

    if px.empty:
        return mark_failed("I14_lny_pre_holiday_drift", "No prices loaded")

    rets = px.pct_change()

    # Build a daily position: 0.5 weight EWH + 0.5 weight EWS in pre-LNY 15
    # trading-day windows; 0 elsewhere.
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    annual_records = []
    for year, dstr in LNY_DATES.items():
        lny = pd.Timestamp(dstr)
        # Find trading day on or just before lny
        idx = rets.index.searchsorted(lny)
        if idx == 0:
            continue
        # LNY-1 is the last trading day strictly before lny
        end_pos = idx - 1
        start_pos = end_pos - 14  # 15-session window
        if start_pos < 0:
            continue
        if end_pos >= len(rets):
            end_pos = len(rets) - 1
        window = rets.index[start_pos:end_pos+1]
        if len(window) == 0:
            continue
        pos.loc[window, "EWH"] = 0.5
        pos.loc[window, "EWS"] = 0.5
        # Per-year captured return (long-only avg of EWH+EWS over the window)
        w_rets = rets.loc[window, ["EWH", "EWS"]].mean(axis=1).fillna(0)
        cum = float((1 + w_rets).prod() - 1)
        annual_records.append({
            "year": year,
            "window_start": str(window[0].date()),
            "window_end": str(window[-1].date()),
            "n_days": int(len(window)),
            "captured_return": cum,
        })

    pnl_each = (pos.shift(1) * rets).sum(axis=1)
    pnl = pnl_each.dropna()

    if pnl.abs().sum() == 0:
        return mark_failed("I14_lny_pre_holiday_drift",
                           "Signal never fired (data alignment problem)")

    # Time-in-market metrics
    days_active = int((pos.sum(axis=1) > 0).sum())
    total_days = int(len(pos))
    timeinmkt = days_active / total_days if total_days > 0 else 0.0

    # Cumulative captured return across all LNY windows
    active_only = pnl[pos.shift(1).sum(axis=1) > 0]
    cum_capture = float((1 + active_only).prod() - 1) if not active_only.empty else 0.0
    annualized_capture = (1 + cum_capture) ** (252.0 / max(days_active, 1)) - 1 if days_active > 0 else 0.0

    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="I-14 LNY pre-holiday EWH/EWS")
    print_metrics(m)
    print(f"\nTime-in-market: {timeinmkt:.2%} ({days_active}/{total_days} days)")
    print(f"Cumulative captured return (active days only): {cum_capture*100:.2f}%")
    print(f"Annualized when active: {annualized_capture*100:.2f}%")
    print("Per-year:")
    for r in annual_records:
        print(f"  {r['year']}: {r['captured_return']*100:+6.2f}%  ({r['n_days']}d)")

    save_result("I14_lny_pre_holiday_drift", m, extra={
        "status": "ok",
        "rule": ("Long 50/50 EWH+EWS from LNY-15 trading sessions through "
                 "LNY-1 session each year."),
        "data_source": "Hardcoded LNY dates; iShares EWH/EWS via yfinance.",
        "time_in_market": float(timeinmkt),
        "cumulative_capture": cum_capture,
        "annualized_when_active": float(annualized_capture),
        "n_years": len(annual_records),
        "per_year": annual_records,
    })


if __name__ == "__main__":
    main()
