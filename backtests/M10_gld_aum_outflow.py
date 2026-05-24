"""
M10 GLD AUM outflow.
Plan: scrape SPDR Gold Shares historical tonnes-in-trust.
Reality: SPDR's "archive CSV" endpoint returns a PDF, the historical-data page
is a JS-rendered React app, and the World Gold Council ETF API requires a
registered key. yfinance and Stooq give GLD price/volume but not AUM/tonnes.

Without tonnes-in-trust data we cannot test the rule cleanly (using GLD price
as a proxy would tautologically correlate with the target return).
Marking failed honestly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "M10_gld_aum_outflow",
        "SPDR holdings CSV endpoint serves a PDF; historical-data page is JS-rendered; WGC API requires key. No clean free source for historical GLD tonnes-in-trust.",
        extra={
            "mechanism": "Capitulation outflows from GLD historically mark gold price bottoms (contrarian).",
            "source_attempted": "spdrgoldshares.com archive CSV (PDF), historical-data page (JS app), gold.org ETF holdings (404).",
        },
    )


if __name__ == "__main__":
    main()
