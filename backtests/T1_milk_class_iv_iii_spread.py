"""
T1 Milk Class IV - Class III spread.

Rule: When (DK - DC) < -$3.50, long DK / short DC; exit at zero crossing, hold up to 6 months.

Data:
  - DC=F (Class III Milk) available on yfinance.
  - DK=F (Class IV Milk) IS NOT available on yfinance (404).

We tried CME free settlement data but settlement endpoints are auth-walled
for the historical archive (CME requires a feed subscription / Data Mine).
Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yfinance as yf
from harness import mark_failed


def main():
    try:
        dk = yf.download("DK=F", start="2010-01-01", progress=False, auto_adjust=True)
        if dk.empty:
            return mark_failed(
                "T1_milk_class_iv_iii_spread",
                "DK=F (CME Class IV Milk) not available on yfinance (404). "
                "DC=F (Class III) works; spread is impossible without paid CME Data Mine.",
                extra={"rule": "spread DK-DC < -$3.50: long DK/short DC, exit at 0",
                       "mechanism": "Class IV (butter/powder) vs Class III (cheese) supply rotation",
                       "source": "CME settlements (subscription/feed)"}
            )
    except Exception as e:
        pass

    return mark_failed(
        "T1_milk_class_iv_iii_spread",
        "DK=F unavailable on yfinance; CME public settlement archive requires Data Mine subscription.",
        extra={"rule": "spread DK-DC < -$3.50: long DK/short DC, exit at 0",
               "mechanism": "Class IV vs Class III dairy product spread",
               "source": "CME Group (DK=F not free)"}
    )


if __name__ == "__main__":
    main()
