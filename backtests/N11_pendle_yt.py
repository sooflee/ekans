"""
N11 Pendle YT supply / implied APY.
Pendle's api-v2.pendle.finance exposes current state but historical YT supply per
market is only retrievable by scraping the app or running an indexer against
their subgraphs. Setting up the indexer for ~50 markets across multiple chains
is beyond this batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N11_pendle_yt",
        "Pendle historical YT supply not exposed via public REST API; requires The Graph subgraph indexer.",
        extra={
            "rule_intended": "Long ETH / yield-token basket when aggregate YT supply expands > +20% wk-over-wk.",
            "source": "api-v2.pendle.finance + Pendle subgraphs",
            "universe": "ETH / Pendle YT markets",
        },
    )
    print("N11 Pendle YT: marked failed (no public historical API).")


if __name__ == "__main__":
    main()
