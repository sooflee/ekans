"""
P7 Constan Treasury Refunding bill-share.

Spec: Treasury Refunding Statements published 1st Wed of Feb/May/Aug/Nov.
Need bill-share % of total marketable issuance. When bill-share > 5y-trailing
avg + 3pp, long TLT+SPY for 1 quarter.

Implementation: We bypass scraping the press release by directly computing
bill share from the Treasury auction history (data/treasury_auctions_all.parquet),
which records actual issuance amounts. For each quarter-end, we compute the
trailing-3-month bill share of new marketable issuance (Bills+Notes+Bonds).
Then we compare to the trailing-5-year quarterly average and trigger on
exceeding avg + 3pp. Position is held for the next quarter (63 trading days).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result,
)


def main():
    auc = pd.read_parquet("/Users/benson/Projects/ekans/data/treasury_auctions_all.parquet")
    auc = auc[auc["security_type"].isin(["Bill", "Note", "Bond"])].copy()
    auc["issue_date"] = pd.to_datetime(auc["issue_date"], errors="coerce")
    auc["total_accepted"] = pd.to_numeric(auc["total_accepted"], errors="coerce")
    auc = auc.dropna(subset=["issue_date", "total_accepted"])
    auc = auc[auc["issue_date"] >= "1995-01-01"]
    auc["q"] = auc["issue_date"].dt.to_period("Q").dt.to_timestamp("Q")
    grp = (auc.groupby(["q", "security_type"])["total_accepted"]
              .sum()
              .unstack(fill_value=0.0))
    grp["total_marketable"] = grp.sum(axis=1)
    grp["bill_share"] = grp.get("Bill", 0.0) / grp["total_marketable"]
    # Trailing 5-year avg of bill share (20 quarters)
    grp["bill_share_5y_avg"] = grp["bill_share"].rolling(20, min_periods=12).mean()
    grp["signal"] = (grp["bill_share"] > grp["bill_share_5y_avg"] + 0.03)

    # Build daily position: signal at quarter-end Q applies to the NEXT quarter
    # (Q+1's trading days), as the refunding announcement comes after Q closes.
    spy = load_prices(["SPY"], start="1995-01-01")["SPY"]
    tlt = load_prices(["TLT"], start="2002-08-01")["TLT"]
    common = spy.index.intersection(tlt.index)
    spy_r = spy.pct_change().reindex(common)
    tlt_r = tlt.pct_change().reindex(common)
    pair_r = 0.5 * spy_r + 0.5 * tlt_r

    # For each (quarter, signal=True), set position 1.0 for the days in the next quarter.
    pos = pd.Series(0.0, index=common)
    n_trig = 0
    for q, row in grp.iterrows():
        if not bool(row["signal"]) or pd.isna(row["bill_share_5y_avg"]):
            continue
        next_q_start = q + pd.Timedelta(days=1)
        next_q_end = next_q_start + pd.offsets.QuarterEnd(0)
        mask = (common >= next_q_start) & (common <= next_q_end)
        pos.loc[common[mask]] = 1.0
        n_trig += 1

    pnl = (pos.shift(1).fillna(0) * pair_r).dropna()
    m = compute_metrics(pnl, benchmark=spy_r.dropna(), name="P7 Constan Bill-Share")
    print_metrics(m)
    save_result("P7_constan_treasury_refunding", m, extra={
        "status": "ok",
        "rule": "If trailing-quarter bill share of Treasury marketable issuance > 5y avg + 3pp, long SPY+TLT (50/50) for next quarter.",
        "mechanism": "Andy Constan: when Treasury skews issuance to bills (front-end) vs coupons, it eases financial conditions; risk-on for stocks+long bonds.",
        "universe": "SPY + TLT pair (50/50); signal: Treasury auction issuance composition.",
        "substitution_notes": "Bypassed press-release scraping by computing bill share directly from Treasury auction history (offering amounts).",
        "n_triggered_quarters": n_trig,
        "n_total_quarters": int(grp['signal'].notna().sum()),
        "source": "Andy Constan, Damped Spring (Phase 1P, YouTube)",
    })


if __name__ == "__main__":
    main()
