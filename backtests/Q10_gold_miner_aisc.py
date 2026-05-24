"""
Q10 Gold miner AISC -> long GLD when spot near AISC.

Would require curating quarterly AISC (all-in sustaining cost) values for ~10 gold
majors (Newmont, Barrick, AngloGold, Kinross, Agnico, Gold Fields, Newcrest, Polyus,
Freeport, Yamana) from SEC 10-Qs and JSE/ASX filings, then weighting by GDX
constituent weights for each quarter. This is heavy data engineering for one signal.

Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q10_gold_miner_aisc",
        "Quarterly AISC values for 10 majors require curation across SEC EDGAR 10-Qs and "
        "non-US filings (JSE, ASX, MOEX). Weighting by GDX constituents (which themselves "
        "change quarterly) and aligning quarters is heavy. Skipped.",
        extra={
            "source_attempted": ["SEC EDGAR 10-Q filings", "Company press releases", "GDX index methodology"],
            "rule_intended": "When spot gold within 10% of GDX-weighted average AISC, long GLD 120 trading days.",
            "mechanism": "Marginal-cost support: gold near miners' AISC induces production cuts and capex deferrals.",
        },
    )


if __name__ == "__main__":
    main()
