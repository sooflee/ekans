"""
L5 NRC Power Reactor Status — accessible but server rate-limits hard.

Format is documented:
  https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/<YEAR>/<YEAR>PowerStatus.txt
  pipe-delimited: ReportDt|Unit|Power, daily snapshots per unit (0-100%).

First request from this network successfully retrieved
2023PowerStatus.txt (34,078 rows). Subsequent requests for additional
years are met with HTTP/2 stream INTERNAL_ERROR or timeouts — the host
applies aggressive per-IP rate limiting / WAF rules to www.nrc.gov.

Without 10+ years of fleet data we cannot construct a meaningful rolling
fleet-power signal. Mark failed rather than ship a single-year backtest
or a flaky scraper that depends on luck against the WAF.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L5_nrc_nuclear_outages",
        ("NRC bulk PowerStatus.txt files are technically downloadable, but "
         "www.nrc.gov rate-limits successive requests with HTTP/2 stream "
         "INTERNAL_ERROR and 60-90s read timeouts after the first hit. "
         "Cannot reliably build a multi-year fleet outage panel from a free "
         "scrape without an allowed harvest agreement."),
        extra={
            "source_attempted": "https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/<YEAR>/<YEAR>PowerStatus.txt",
            "rule": ("7d mean nuclear fleet power < 80% -> long XLU 10 sessions "
                     "(intended rule; not testable here)."),
            "mechanism": "Elevated nuclear outages shift load to gas peakers; lifts utility realized power prices.",
        },
    )


if __name__ == "__main__":
    main()
