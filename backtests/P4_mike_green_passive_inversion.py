"""
P4 Mike Green Passive-Flow Inversion.

Goal: Use ICI long-term equity mutual fund flow data to detect 3-month
rolling-flow turning negative; when it does, reduce SPY exposure (go to cash)
for 60 days.

Status: ICI's public stats pages (https://www.ici.org/research/stats) gate
historical CSVs behind login / non-machine-readable HTML tables; the
documented URL endpoints return 404 for direct CSV access (verified
2026-05-24). No FRED mirror of the ICI long-term equity mutual fund
flow series exists.

Marking as fail (data-availability). Possible future workaround: scrape the
ICI Weekly Estimated Long-Term Mutual Fund Flows PDF/XLS table programmatically
or pay for the ICI data feed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed("P4_mike_green_passive_inversion",
                "ICI long-term mutual fund flow CSVs are not publicly machine-readable; FRED has no mirror. Verified ici.org endpoints return 404 or HTML-only.",
                extra={
                    "rule": "Long SPY normally; when ICI 3-month equity mutual fund net flow turns negative, go to cash for 60 days.",
                    "mechanism": "Mike Green: passive inflows are the marginal price-setter; when they reverse, mechanical support disappears.",
                    "source": "Mike Green, Simplify (YouTube/podcasts, Phase 1P)",
                    "data_gap": "ICI Weekly/Monthly Long-Term Mutual Fund flow series not available via FRED or as a free public CSV.",
                })


if __name__ == "__main__":
    main()
