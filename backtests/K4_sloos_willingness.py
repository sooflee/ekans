"""
K4 SLOOS Willingness to Make Consumer Installment Loans (FRED DRIWCIL)

Rule: When QoQ change in DRIWCIL > +10 (net percent), go long XLY (consumer
discretionary) for 60 trading days. Otherwise flat.

Mechanism: A sharp jump in banks' willingness to extend consumer credit is a
leading indicator of household spending growth, which directly benefits
consumer-discretionary names. The Senior Loan Officer Opinion Survey is
released quarterly with a ~1 week lag from the start of each quarter — we
shift the signal date forward by 1 month to be safe (lag after publication).

Honest notes:
 - Quarterly observations; ~30 years of data → modest sample.
 - We treat each trigger as starting at the publication-date-plus-lag and
   running for 60 trading days; overlapping triggers are averaged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, load_fred, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    try:
        sloos = load_fred(["DRIWCIL"], start="1995-01-01")
    except Exception as e:
        return mark_failed("K4_sloos_willingness", f"FRED DRIWCIL fetch failed: {e}")

    s = sloos["DRIWCIL"].dropna()
    s = s.sort_index()
    qoq = s.diff()
    # Publication lag: SLOOS released ~5 weeks after quarter start. Add 35 days.
    pub_dates = s.index + pd.Timedelta(days=35)
    qoq.index = pub_dates

    try:
        px = load_prices(["XLY", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed("K4_sloos_willingness", f"price fetch: {e}")

    rets = daily_returns(px)
    xly_r = rets["XLY"].dropna()

    # Build signal: 1 if QoQ > 10, held 60 trading days from the publication date
    sig = pd.Series(0.0, index=xly_r.index)
    n_triggers = 0
    for ts, val in qoq.dropna().items():
        if val > 10:
            start_idx = xly_r.index.searchsorted(ts)
            if start_idx >= len(xly_r):
                continue
            end_idx = min(start_idx + 60, len(xly_r))
            sig.iloc[start_idx:end_idx] += 1.0
            n_triggers += 1

    # Cap position at 1 (overlap → still long, no leverage)
    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * xly_r
    pnl = pnl.dropna()

    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K4 SLOOS willingness long XLY")
    print_metrics(m)
    print(f"n_triggers={n_triggers}, days_in_market={int(pos.sum())}/{len(pos)}")

    save_result("K4_sloos_willingness", m, extra={
        "status": "ok",
        "rule": "When QoQ change in DRIWCIL > +10, long XLY for 60 trading days. Overlaps stay long.",
        "mechanism": "Bank willingness to lend leads consumer discretionary spending.",
        "source": "FRED DRIWCIL (SLOOS Willingness to Make Consumer Installment Loans, quarterly)",
        "n_triggers": int(n_triggers),
        "days_in_market": int(pos.sum()),
        "exposure_pct": float(pos.mean()),
    })


if __name__ == "__main__":
    main()
