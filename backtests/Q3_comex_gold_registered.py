"""
Q3 COMEX gold registered stocks.

CME publishes daily 'Metals Issues and Stops' reports on cmegroup.com. The historical
archive is not exposed as a clean machine-readable feed; reports are PDF/HTML and
require day-by-day scraping. Wayback coverage is sparse.

Marking failed - dependency on heavy scraping and no clean alternative free series.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q3_comex_gold_registered",
        "CME 'Metals Issues and Stops' daily reports are PDF/HTML only; no free clean "
        "historical timeseries. Wayback coverage is incomplete and per-day scraping for "
        "10+ years of daily PDFs is out of scope.",
        extra={
            "source_attempted": [
                "https://www.cmegroup.com/market-data/reports/metals-issues-and-stops-report.html",
                "https://web.archive.org/web/*/cmegroup.com/market-data/reports/metals-issues-and-stops-report.html",
            ],
            "rule_intended": "When COMEX gold registered stocks drop below 5y rolling min, long GLD.",
            "mechanism": "Falling registered stocks indicate physical squeeze risk; lifts spot gold premium.",
        },
    )


if __name__ == "__main__":
    main()
