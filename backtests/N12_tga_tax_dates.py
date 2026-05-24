"""
N12 TGA tax-date pump.
Hardcoded estimated-tax dates: Apr 15, Jun 15, Sep 15, Jan 15 from 2018-2026.
For each event, check if FRED WTREGEN (Treasury General Account, weekly) rose > $80B
in the preceding 5 weeks. If so, long TLT for 5 sessions starting on day-after-tax-date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        # WTREGEN: Treasury General Account weekly (Wednesday close, $ millions)
        tga = load_fred("WTREGEN", start="2015-01-01").iloc[:, 0].rename("TGA")
        tlt = load_prices(["TLT"], start="2015-01-01").iloc[:, 0].rename("TLT")
    except Exception as e:
        return mark_failed("N12_tga_tax_dates", f"data load failed: {e}")

    # Build tax-date list
    tax_dates = []
    for y in range(2018, 2027):
        for m, d in [(1, 15), (4, 15), (6, 15), (9, 15)]:
            tax_dates.append(pd.Timestamp(year=y, month=m, day=d))

    tlt_rets = tlt.pct_change()
    pos = pd.Series(0.0, index=tlt_rets.index)
    events = []
    for td in tax_dates:
        # find TGA value within +/- 7 days of tax date (latest weekly point <= tax date)
        prior = tga.loc[:td]
        if prior.empty:
            continue
        tga_now = prior.iloc[-1]
        # 5 weeks earlier
        ref_date = td - pd.Timedelta(weeks=5)
        prior_ref = tga.loc[:ref_date]
        if prior_ref.empty:
            continue
        tga_then = prior_ref.iloc[-1]
        rise_bn = (tga_now - tga_then) / 1000.0  # WTREGEN units: $ millions
        if rise_bn <= 80.0:
            continue
        # event qualifies: long TLT for 5 sessions starting day after td
        loc = tlt_rets.index.searchsorted(td)
        for k in range(1, 6):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0
        events.append((td.date().isoformat(), float(rise_bn)))

    if not events:
        return mark_failed("N12_tga_tax_dates", "no qualifying events under $80B/5wk rise threshold",
                           extra={"n_dates_checked": len(tax_dates)})

    pnl_full = (pos * tlt_rets).dropna()
    pnl = pnl_full.loc[pnl_full.ne(0).cummax()]
    bench = tlt_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N12 TGA tax-date long TLT")
    m["n_events"] = len(events)
    print(f"Qualifying events: {len(events)}/{len(tax_dates)}")
    for e in events[:6]:
        print(" ", e)
    print_metrics(m)
    save_result("N12_tga_tax_dates", m, extra={
        "status": "ok",
        "rule": "If WTREGEN rose > $80B in 5 weeks before estimated-tax due date (Apr/Jun/Sep/Jan 15), long TLT 5 sessions starting day-after.",
        "mechanism": "Treasury cash drain forces refunding; rates rally as bills mature/coupons fall",
        "universe": "TLT",
        "source": "Treasury DTS / FRED WTREGEN",
        "events": events,
    })


if __name__ == "__main__":
    main()
