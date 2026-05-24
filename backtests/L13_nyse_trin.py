"""
L13 NYSE TRIN -- Stooq gated behind API key; yfinance has no ^TRIN.

Stooq (the documented free source for ^TRIN) now requires a per-account
API key obtained via captcha before any CSV download. yfinance returns
404 for ^TRIN / ^TICK / ^ADV / ^DECL / ^UVOL / ^DVOL (those symbols are
not on Yahoo's quote service).

A manual TRIN reconstruction needs NYSE advances/declines/up-volume/
down-volume — also not available free at daily granularity for 20+
years.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L13_nyse_trin",
        ("Stooq ^TRIN now requires an API key (captcha registration). "
         "yfinance returns 404 for ^TRIN / ^TICK / ^ADV / ^DECL. No free "
         "20+ year daily TRIN feed found."),
        extra={
            "rule": ("TRIN close > 2.5 -> long SPY 5 sessions (intended rule; not testable here)."),
            "mechanism": "Classic capitulation indicator -- extreme down-volume vs. declining issues marks short-term bottoms.",
            "source_attempted": "https://stooq.com/q/d/l/?s=^trin (now apikey-gated).",
        },
    )


if __name__ == "__main__":
    main()
