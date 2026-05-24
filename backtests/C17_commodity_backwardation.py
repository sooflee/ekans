"""
C17 Commodity backwardation
Requires futures curve data (front month vs deferred contracts) not available on free APIs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "C17_commodity_backwardation",
        "Front/back commodity futures curves are not available on yfinance/FRED. Building a real "
        "backwardation/contango signal requires per-commodity continuous front vs second-month series "
        "(or 12m-spread series) which sit behind paid feeds (Bloomberg, Quandl Pro, CME DataMine). "
        "Cite Erb & Harvey 'The Strategic and Tactical Value of Commodity Futures' (FAJ 2006).",
        extra={
            "status": "fail",
            "rule": "Long backwardated commodities, short contangoed ones; monthly rebalance.",
            "universe": "Commodity futures basket",
            "source": "Erb-Harvey FAJ 2006; Gorton-Rouwenhorst 2006",
        }
    )


if __name__ == "__main__":
    main()
