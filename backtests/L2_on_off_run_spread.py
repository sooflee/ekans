"""
L2 On/Off-the-run Treasury spread — requires CUSIP-level data, no free source.

The on/off-the-run premium is computed from yield differences between the most
recently auctioned (on-the-run) and seasoned (off-the-run) issues of the same
tenor. Constructing this cleanly requires per-CUSIP daily yield curves
(Bloomberg ULAR, JPMaQS, GovPX, or NYFRB SOMA data joined with CUSIP-level
quotes). None of these are available via a free public API.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L2_on_off_run_spread",
        "On/off-the-run spread requires per-CUSIP Treasury yield panel; not available free.",
        extra={
            "rule": ("On/off-run 10Y spread > 2σ above 1y mean -> long IEF (premium captures "
                     "liquidity stress) -- intended rule, not testable."),
            "mechanism": "Widening on/off spread signals funding/liquidity stress; subsequent flight-to-quality bid for Treasuries.",
            "source_attempted": "Would need Bloomberg/JPMaQS/GovPX CUSIP yields.",
        },
    )


if __name__ == "__main__":
    main()
