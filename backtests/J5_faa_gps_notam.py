"""
J5 FAA GPS NOTAM. The FAA NOTAM Search API requires registration and
its historical NOTAM text is not freely accessible. Marking as failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed("J5_faa_gps_notam",
                "FAA NOTAM historical text not free-API accessible (external-api.faa.gov requires registered API client; bulk NOTAM history is not published as a free time series).",
                extra={
                    "rule": "(skipped) GPS-NOTAM frequency surge -> short defense/aviation.",
                    "mechanism": "GPS-jamming NOTAM bursts proxy geopolitical risk.",
                    "source": "https://external-api.faa.gov/notamapi (registration required)",
                    "n_events": 0,
                })


if __name__ == "__main__":
    main()
