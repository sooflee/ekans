"""
K1 AIA Inquiries Index proxy via BLS Architectural Services payroll.

Honest substitution: AIA does not publish the ABI/Inquiries Index in
machine-readable historical form (its public page lists only the latest
monthly press release; a "historical data" search returns 404). We use
the BLS series CES6054130001 (All Employees, Architectural, Engineering
and Related Services, SA, monthly via FRED) as a proxy for architecture
activity. The AIA rule "Inquiries > 55 → long XHB 20d" cannot map cleanly
to a payroll level, so we translate to:

  Rule (proxy): when the 3-month change in CES6054130001 prints above its
  trailing 5-year 80th percentile, go long XHB for 20 trading days
  starting the BLS release day (~1st Friday of the month after the data
  month).

Mechanism: rising architecture/engineering employment is a leading
indicator of construction/home-build activity that flows to homebuilder
margins.

Caveat: this is a proxy. The original AIA Inquiries series is paywalled
in machine-readable form, so this is the closest publicly-replicable
substitute.
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
        df = load_fred(["CES6054130001"], start="1995-01-01")
    except Exception as e:
        return mark_failed("K1_aia_inquiries", f"FRED CES6054130001: {e}")
    s = df["CES6054130001"].dropna().astype(float).sort_index()
    mom3 = s.diff(3)

    try:
        px = load_prices(["XHB", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed("K1_aia_inquiries", f"price fetch: {e}")
    rets = daily_returns(px)
    xhb = rets["XHB"].dropna()

    # 5-year rolling 80th percentile (60 months)
    threshold = mom3.rolling(60).quantile(0.80)

    sig = pd.Series(0.0, index=xhb.index)
    n_trig = 0
    for ts, val in mom3.dropna().items():
        thr = threshold.loc[ts] if ts in threshold.index else np.nan
        if pd.isna(thr):
            continue
        if val > thr:
            # BLS payroll for month T is released first Friday of T+1.
            pub = ts + pd.offsets.MonthEnd(0) + pd.offsets.Day(7)
            i = xhb.index.searchsorted(pub)
            if i >= len(xhb):
                continue
            sig.iloc[i:i+20] += 1
            n_trig += 1
    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * xhb
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K1 BLS arch-eng proxy long XHB")
    print_metrics(m)
    print(f"obs={len(s)}, n_trig={n_trig}, exposure={pos.mean():.2%}")

    save_result("K1_aia_inquiries", m, extra={
        "status": "ok_proxy",
        "rule": "3-month change in CES6054130001 above trailing 5y 80th pct → long XHB 20 trading days from BLS release date.",
        "mechanism": "Architecture/engineering payroll growth leads construction & home-build activity.",
        "source": "FRED CES6054130001 (BLS — proxy for AIA ABI, which is not freely machine-readable).",
        "n_observations": int(len(s)),
        "n_triggers": int(n_trig),
        "exposure_pct": float(pos.mean()),
        "caveats": "Original AIA Inquiries Index not freely available in machine-readable form; this is a proxy.",
    })


if __name__ == "__main__":
    main()
