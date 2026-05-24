"""
S6 LME aluminum stock drawdown -> long AA / CENX.

Daily LME on-warrant aluminum stocks: historical archives are behind
LME / FastMarkets / Argus paywalls.

Attempted free sources:
- lme.com/Market-Data/Reports-and-data/Stocks -> 403 Forbidden.
- westmetall.com (mirrors LME) -> only the *current* day snapshot is
  shown on the public table; historical CSV requires login.
- stooq lmal.f / lmar.f -> empty response.
- shanghaimetal.com aluminum-stocks-historical -> 404.

The rule requires DAILY on-warrant LME aluminum stocks back ~10 years
to compute a rolling 8-week % change. No free archival source was found.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "S6_lme_aluminum",
        "LME daily aluminum stock historical archive requires a paid LME/FastMarkets subscription. "
        "Public LME endpoints return 403; westmetall mirror only exposes the current-day snapshot.",
        extra={
            "rule": "When LME on-warrant aluminum stocks fall > 25% in a rolling 8-week window, long AA / CENX for 2 months.",
            "mechanism": "Sharp warehouse drawdowns signal physical market tightness, pulling LME aluminum prices higher and boosting U.S. aluminum equities (Alcoa, Century).",
            "source_attempted": "lme.com/Market-Data/Reports-and-data/Stocks (403); westmetall.com (current snapshot only); stooq (empty).",
        },
    )


if __name__ == "__main__":
    main()
