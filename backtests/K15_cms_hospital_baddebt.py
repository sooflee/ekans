"""
K15 CMS Hospital Cost Reports — quarterly bad-debt time series.

Status: marked failed.

CMS Hospital Cost Reports (Form CMS-2552-10) are published as annual,
provider-level flat files (HOSPITAL10_alpha + HOSPITAL10_nmrc + HOSPITAL10_rpt)
in a multi-table relational format. Extracting "bad debt" line items (Worksheet
S-10, lines 27-30) requires:
 1. Downloading multi-GB annual zip files from cms.gov,
 2. Joining the row, column, and worksheet codes across 3-4 tables,
 3. Mapping worksheet codes to line items (codes change occasionally),
 4. Aggregating to a quarterly time series.

This is significant ETL work outside the scope of a single-pass backtest
script. CMS does not publish a clean quarterly bad-debt time series.

Marked failed per the K-batch instruction: "Complex CSV from CMS. Likely
too heavy. mark_failed with cite if so."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed("K15_cms_hospital_baddebt",
                "CMS Hospital Cost Reports (Form 2552-10) are multi-GB relational flat files "
                "(HOSPITAL10_alpha/nmrc/rpt). Extracting Worksheet S-10 bad-debt lines requires "
                "a multi-step row/column/worksheet code join across 3+ tables for each annual "
                "release, plus careful normalization for fiscal-year reporting. CMS publishes no "
                "clean quarterly bad-debt aggregate. Out of scope for a single-script backtest.",
                extra={"source_checked": "cms.gov/data-research/statistics-trends-and-reports/cost-reports/cost-reports-form-cms-2552-10"})


if __name__ == "__main__":
    main()
