"""
T2 Boxed beef cutout / cattle ratio -> short live cattle.

Rule: cutout/cattle ratio < 1.55 -> short LE=F, hold 30 days.

Data paths tried:
  - USDA AMS MARS API (marsapi.ams.usda.gov): 403 - requires registered API key.
  - mymarketnews.ams.usda.gov public endpoints: connection timeouts.
  - www.ams.usda.gov/mnreports/lm_xb459.txt: serves only the LATEST report,
    no historical archive (and is "Report delayed due to technical difficulties").

To do this properly you would need to register for the MARS API key (free
but requires email approval) and pull each LM_XB459 and LM_CT150 report
through their date-ranged endpoint. That is too heavy for this batch.

Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "T2_boxed_beef_cattle",
        ("USDA AMS LM_XB459 (boxed beef cutout) and LM_CT150 (cattle) "
         "historical archives require MARS API key (free but email-approval "
         "gated). Public ams.usda.gov text files serve only the latest report, "
         "no history."),
        extra={
            "rule": "cutout/cattle ratio < 1.55 -> short LE=F, hold 30d",
            "mechanism": "Margin compression in packer -> cattle softer in lag",
            "source": "USDA AMS LM_XB459 + LM_CT150 (MARS API key required)",
        }
    )


if __name__ == "__main__":
    main()
