"""
Q6 Brent-Dubai EFS (Exchange of Futures for Swaps) settlement.

ICE Brent-Dubai EFS is a sour-sweet differential traded on ICE/CME. ICE only exposes
this via paid market data. There is no free CSV with sufficient history. Without the
Dubai leg this signal can't be evaluated as designed.

Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q6_brent_dubai_efs",
        "ICE Brent-Dubai EFS settlement requires paid ICE market data. The Dubai leg "
        "has no clean free series on yfinance/FRED, and skipping it deflates the signal "
        "to plain Brent (no information advantage).",
        extra={
            "source_attempted": ["https://www.ice.com (ICE Brent-Dubai EFS, paid)"],
            "rule_intended": "EFS spread regimes drive long Brent or long Dubai trades.",
            "mechanism": "Brent-Dubai narrow = Asian heavy/sour demand; wide = Atlantic glut.",
        },
    )


if __name__ == "__main__":
    main()
