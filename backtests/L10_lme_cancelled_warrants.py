"""
L10 LME cancelled warrants — historical warehouse-level data is paid.

LME publishes daily stock and warrant reports (live snapshots) on
lme.com under "Stocks reports", but historical archives at the warehouse
location level (which the cancelled-warrant signal needs) are restricted
to LMEselect / FastMarkets data subscribers. The free PDF stock reports
expire after a short rolling window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L10_lme_cancelled_warrants",
        "LME warehouse cancelled-warrant historical series is not available "
        "via a free public API (paid LME data subscription required).",
        extra={
            "rule": ("Cancelled warrants > 40% of on-warrant stocks for a metal -> "
                     "long the metal ETF (e.g. JJC for copper) 10 sessions -- not testable here."),
            "mechanism": "High cancelled-warrants ratio signals imminent physical withdrawal; pressures cash-3m spreads tighter and lifts price.",
            "source_attempted": "https://www.lme.com/Market-Data/Reports-and-data",
        },
    )


if __name__ == "__main__":
    main()
