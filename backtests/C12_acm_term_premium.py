"""
C12 ACM 10Y term premium
FRED THREEFYTP10 (Adrian-Crump-Moench). Rule: when ACM 10Y rises >50bps from 12-month low
and crosses zero positive, short TLT or shift to cash for 6 months.
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
        acm = load_fred("THREEFYTP10", start="2002-07-30").iloc[:, 0].rename("ACM")
        tlt = load_prices(["TLT"], start="2002-07-30").iloc[:, 0].rename("TLT")
    except Exception as e:
        return mark_failed("C12_acm_term_premium", f"data load failed: {e}")

    df = pd.DataFrame({"ACM": acm}).reindex(tlt.index, method="ffill")
    df["TLT"] = tlt
    df = df.dropna()

    # 12 month rolling min ~ 252 trading days
    rmin = df["ACM"].rolling(252).min()
    risen_50bps = (df["ACM"] - rmin) >= 0.5
    prev_acm = df["ACM"].shift(1)
    crossed_pos = (prev_acm <= 0) & (df["ACM"] > 0)
    trigger = risen_50bps & crossed_pos

    # 6-month (126 trading day) short TLT after trigger
    pos = pd.Series(0.0, index=df.index)
    countdown = 0
    for i in range(len(df)):
        if trigger.iloc[i]:
            countdown = 126
        if countdown > 0:
            pos.iloc[i] = -1.0
            countdown -= 1

    tlt_ret = df["TLT"].pct_change()
    pnl = (pos.shift(1) * tlt_ret).dropna()
    bench = tlt_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C12 ACM 10Y term premium")
    print_metrics(m)
    save_result("C12_acm_term_premium", m, extra={
        "status": "ok",
        "rule": "Short TLT for 6 months when ACM10Y rises >=50bps from 12mo low AND crosses 0 from below.",
        "universe": "TLT (short)",
        "source": "Adrian-Crump-Moench (NY Fed) term premium series THREEFYTP10",
    })


if __name__ == "__main__":
    main()
