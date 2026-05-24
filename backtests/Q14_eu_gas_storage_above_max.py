"""
Q14 EU gas storage above 5y same-week max.

AGSI+ daily storage data is freely available via REST API (gie.eu / agsi.eu/data),
but the trade leg requires TTF natural gas futures which are not on yfinance.
US natgas (UNG / NG=F) is a poor proxy for TTF (different supply/demand).

Marking failed - TTF futures not freely available.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q14_eu_gas_storage_above_max",
        "TTF natural gas futures are not available on yfinance/FRED. UNG (US natgas) is "
        "not a meaningful proxy for European gas (different basis). AGSI+ storage data is "
        "available but unusable without a TTF return series.",
        extra={
            "source_attempted": ["AGSI+ (gie.eu, available)", "TTF futures (paid only)"],
            "rule_intended": "EU aggregate fill > 5y same-week max by >5pp -> short TTF for 30d.",
            "mechanism": "Glut storage caps near-term TTF price.",
        },
    )


if __name__ == "__main__":
    main()
