"""
Q15 JKM-TTF spread.

JKM (Japan-Korea Marker) and TTF (Dutch) gas futures are paid market data only;
both unavailable on yfinance/FRED. UNG vs UNG is not meaningful.

Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q15_jkm_ttf",
        "JKM and TTF futures are not freely available on yfinance/FRED. No reasonable "
        "proxy exists (UNG is US Henry Hub, structurally different). Skipped.",
        extra={
            "source_attempted": ["ICE JKM (paid)", "EEX TTF (paid)"],
            "rule_intended": "Trade JKM-TTF mean reversion when spread > 2 std historical.",
            "mechanism": "LNG cargo arbitrage between Asia and Europe.",
        },
    )


if __name__ == "__main__":
    main()
