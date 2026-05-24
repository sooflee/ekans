"""
T10 Tampa vs Black Sea urea spread -> long ZC=F.

Rule: Tampa Urea - Black Sea Urea > $80/MT for 3 weeks -> long Dec ZC=F 2-4 months.

Data path tried:
  - World Bank Pink Sheet (Monthly Prices): we can pull the latest xlsx from
    thedocs.worldbank.org/.../CMO-Historical-Data-Monthly.xlsx — it works.
  - HOWEVER: Pink Sheet has ONLY ONE urea series ("Urea " = UREA_EE_BULK,
    which is FOB Black Sea/Baltic, $/mt). There is no separate Tampa urea
    column. Tampa urea (Argus/Profercy) is a paid wire-service price.
  - Argus / Fertecon / Profercy daily Tampa urea spot is subscription-only.

Marking failed: the spread cannot be computed without Tampa data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "T10_tampa_urea",
        ("World Bank Pink Sheet only carries one urea price (UREA_EE_BULK, "
         "FOB Black Sea/Baltic). Tampa urea quotes are from Argus/Fertecon/"
         "Profercy paywalled wire services. Spread cannot be computed from "
         "free data."),
        extra={
            "rule": "Tampa - Black Sea urea > $80/MT for 3 wks -> long ZC=F 2-4mo",
            "mechanism": ("Tampa premium reflects North American gas-cost squeeze; "
                          "implies corn input cost shock -> grain bid"),
            "source": ("World Bank Pink Sheet (Black Sea only) + "
                       "Argus/Fertecon Tampa (paywalled)"),
        }
    )


if __name__ == "__main__":
    main()
