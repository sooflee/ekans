"""
L11 USPTO weekly grants — bulk XML too heavy; old PatentsView API retired.

Two free paths existed and both are now blocked:

(a) The legacy PatentsView REST API
    https://api.patentsview.org/patents/query
    has been replaced by USPTO's Open Data Portal (ODP); the old URLs now
    redirect to the ODP single-page app and the documented endpoint paths
    return HTML, not JSON.

(b) USPTO weekly grant XML at bulkdata.uspto.gov is multi-gigabyte per
    week. Even just parsing 10 years of weekly assignments to extract
    counts per public-company assignee is many TB of decompressed data.

A partial implementation using a small hard-coded assignee list (e.g.
Apple, MSFT, GOOG, etc.) is feasible only with the new ODP API once
USPTO publishes stable JSON endpoints. Marking failed to avoid shipping
a broken prototype.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L11_uspto_grants",
        ("Legacy PatentsView API retired; new USPTO Open Data Portal endpoints "
         "are SPA-bound (no stable JSON). Bulk USPTO weekly XML is too heavy "
         "(GB-scale per week) to ship in a free-tier backtest."),
        extra={
            "rule": ("3 consec weeks of >2σ above 52w avg patent grants for a "
                     "company -> long stock 3 months (intended rule; not testable here)."),
            "mechanism": "Sustained patent grant surges signal R&D output / moat investment paying off.",
            "source_attempted": "PatentsView v0 API (retired); bulkdata.uspto.gov (too heavy).",
        },
    )


if __name__ == "__main__":
    main()
