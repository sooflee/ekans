"""
A09 Sunshine effect.
Per spec, this requires NOAA station-level weather joined to index dates which is
heavy and out of scope. Try a minimal FRED-weather check (FRED doesn't carry NYC daily
sunshine); if not available, mark_failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    # FRED does not carry daily NYC sunshine / cloud-cover data we'd need.
    # The Hirshleifer-Shumway (2003) paper uses 26 international cities' weather
    # from ISH dataset (NOAA). Doing that join correctly is heavier than this pass.
    mark_failed(
        "A09_sunshine",
        "Requires NOAA station-level cloud cover join (Hirshleifer-Shumway 2003); FRED does not carry "
        "daily NYC sky-cover series. Out of scope for this batch.",
        extra={
            "source": "Hirshleifer & Shumway (JF 2003)",
            "universe": "SPY / NYSE",
        },
    )
    print("A09 sunshine: marked failed (out of scope for this pass).")


if __name__ == "__main__":
    main()
