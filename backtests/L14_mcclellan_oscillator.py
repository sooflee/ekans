"""
L14 NYSE McClellan oscillator -- needs ^ADV/^DEC; sources gated.

Stooq ^ADV/^DEC are now apikey-gated (same lockdown that affects ^TRIN).
yfinance returns 404 for ^ADV / ^ADVN / ^DECL / ^DECN. The McClellan
oscillator [19d EMA(A-D) - 39d EMA(A-D)] cannot be reproduced without
those breadth series.

A workable proxy could be built from the SPX constituent set (compute
A-D using S&P 500 components only), but that is a different signal
(not the canonical NYSE McClellan) and would require its own validation
pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L14_mcclellan_oscillator",
        ("NYSE breadth (^ADV / ^DEC) is no longer accessible via free "
         "Stooq endpoints (API key required) or yfinance (404). McClellan "
         "oscillator cannot be reconstructed without these series."),
        extra={
            "rule": ("McClellan oscillator crosses from <-50 up through 0 -> long SPY 21d "
                     "(intended rule; not testable here)."),
            "mechanism": "Breadth thrust off oversold readings tends to precede multi-week SPX rallies.",
            "source_attempted": "Stooq ^ADV/^DEC (apikey-gated); yfinance (404).",
        },
    )


if __name__ == "__main__":
    main()
