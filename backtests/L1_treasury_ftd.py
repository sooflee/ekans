"""
L1 DTCC Treasury Fails-to-Deliver — too heavy to scrape reliably.

DTCC's treasury fails charts at https://www.dtcc.com/charts/treasury-fails-charts
render via JavaScript / live AJAX calls into their datawarehouse, and there is
no documented free-tier API that returns the underlying weekly time series.
Historical CSV downloads behind that page require an authenticated DTCC
Learning Center / Data Products account.

Mark failed rather than risk a brittle scraper that will silently break.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L1_treasury_ftd",
        "DTCC Treasury Fails-to-Deliver historical weekly series is not "
        "available via a free public API. The dtcc.com chart page requires "
        "JS-rendered AJAX, and bulk CSV/Excel downloads need a DTCC Data "
        "Products entitlement.",
        extra={
            "source_attempted": "https://www.dtcc.com/charts/treasury-fails-charts",
            "rule": ("Weekly Treasury FTDs spike (>2σ above 26w mean) -> short TLT 10 sessions "
                     "(intended rule; not testable here)."),
            "mechanism": "Liquidity stress in Treasury market raises term-premium and pressures bond ETFs.",
        },
    )


if __name__ == "__main__":
    main()
