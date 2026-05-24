"""
I-6 Lunar January effect — A-shares.

Hold ASHR (CSI 300 ETF) long ONLY during lunar month 1 (LNY day through ~30
days later). Flat the rest of the year. Compare to buy-and-hold ASHR.

ASHR began trading 2013-11-06. Window is short but interesting.

Lunar month 1 = LNY date through next new moon, ~29-30 days. We use 30
calendar days as a simple approximation.
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
        px = load_prices(["ASHR"], start="2013-11-01")
    except Exception as e:
        return mark_failed("I06_lunar_january_ashares", f"Price load failed: {e}")

    if px.empty:
        return mark_failed("I06_lunar_january_ashares", "ASHR price load failed")

    ashr = px["ASHR"].dropna()
    rets = ashr.pct_change()

    pos = pd.Series(0.0, index=ashr.index)
    annual_records = []
    for year, dstr in LNY_DATES.items():
        lny = pd.Timestamp(dstr)
        end = lny + pd.Timedelta(days=30)
        # window mask
        mask = (rets.index >= lny) & (rets.index <= end)
        if mask.sum() == 0:
            continue
        pos.loc[mask] = 1.0
        w_rets = rets.loc[mask].fillna(0)
        cum = float((1 + w_rets).prod() - 1)
        annual_records.append({
            "year": year,
            "lny": dstr,
            "n_days": int(mask.sum()),
            "lunar_jan_return": cum,
        })

    pnl = (pos.shift(1) * rets).dropna()

    if pnl.abs().sum() == 0:
        return mark_failed("I06_lunar_january_ashares",
                           "Signal never fired")

    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="I-6 Lunar Jan ASHR")
    print_metrics(m)
    days_active = int((pos > 0).sum())
    print(f"\nTime-in-market: {days_active / len(pos):.2%} ({days_active}/{len(pos)} days)")
    cum_capture = float((1 + pnl[pos.shift(1) > 0]).prod() - 1) if days_active > 0 else 0.0
    print(f"Cumulative captured (active days only): {cum_capture*100:+.2f}%")
    print("Per-year:")
    for r in annual_records:
        print(f"  {r['year']}: {r['lunar_jan_return']*100:+6.2f}%  ({r['n_days']}d)")

    save_result("I06_lunar_january_ashares", m, extra={
        "status": "ok",
        "rule": ("Long ASHR for ~30 calendar days starting LNY each year; "
                 "flat otherwise."),
        "data_source": "Hardcoded LNY dates; ASHR via yfinance (since 2013).",
        "time_in_market": float(days_active / len(pos)) if len(pos) else 0.0,
        "cumulative_capture": cum_capture,
        "n_years": len(annual_records),
        "per_year": annual_records,
    })


if __name__ == "__main__":
    main()
