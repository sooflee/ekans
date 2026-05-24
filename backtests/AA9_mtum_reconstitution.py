"""
AA9 MTUM semi-annual reconstitution.
iShares MTUM tracks MSCI USA Momentum Index. Semi-annual rebal: last business
day of May + November.

Spec: compute projected deletions from current MTUM holdings vs the MSCI USA
Momentum methodology (12m-1m ann. risk-adjusted return, top quintile by
sector-relative score). Short projected deletions T-5 to T-0.

This requires:
  1. Current MTUM holdings file (~125 names) — available from iShares CSV
     but per-rebalance historical snapshots aren't free.
  2. MSCI USA universe (~600 names) — proprietary.
  3. Recomputing the momentum methodology to project deletions.

Out of scope for this pass. We mark failed and note the live implementation
path: parse iShares MTUM holdings monthly snapshots from Wayback Machine,
intersect with subsequent month's holdings to identify deletions, then
backtest those names short T-5 to T-0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "AA9_mtum_reconstitution",
        "Requires historical MTUM holdings snapshots × MSCI USA universe × momentum methodology — out of scope.",
        extra={
            "source": "iShares MTUM holdings CSV (current only via iShares.com); MSCI USA Momentum methodology",
            "universe": "Projected MTUM deletions",
            "rebal_dates": "Last business day of May + November",
            "live_path": "Scrape Wayback Machine snapshots of iShares MTUM holdings CSV monthly; diff to get adds/drops; backtest drops short T-5..T-0.",
        },
    )
    print("AA9 MTUM reconstitution: marked failed (curation too heavy).")


if __name__ == "__main__":
    main()
