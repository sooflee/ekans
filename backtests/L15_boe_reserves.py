"""
L15 BoE weekly reserves — IADB CSV endpoint blocked / returns error pages.

The Bank of England Interactive Database (IADB) CSV export URL pattern
documented in their help pages:
  https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp
  ?CSVF=TT&SeriesCodes=<CODE>&UsingCodes=Y&Datefrom=...&Dateto=...
returns either:
  - an HTML error/landing page (the Akamai-wrapped portal redirects browser
    UAs to the SPA), OR
  - a 12 KB JS-laden HTML stub that never resolves to CSV.

Probed candidate weekly reserve codes RPMB17R / RPWB55A / LPMB94B /
LPMBC4N / RPWB55B / RPMA17R all returned the same HTML stub. FRED has no
UK central-bank weekly reserve proxy.

Mark failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L15_boe_reserves",
        ("BoE IADB CSV endpoint returns HTML stubs for all probed reserve "
         "series codes (the portal serves a SPA; programmatic CSV requires "
         "manual browser session / cookies). No FRED proxy for weekly UK "
         "CB reserves."),
        extra={
            "rule": ("BoE reserves decline 4 consecutive weeks -> short GBP/USD via short "
                     "FXB 10 sessions (intended rule; not testable here)."),
            "mechanism": "Sustained BoE reserve drawdowns signal balance-sheet tightening / sterling intervention pressure.",
            "source_attempted": "https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp",
        },
    )


if __name__ == "__main__":
    main()
