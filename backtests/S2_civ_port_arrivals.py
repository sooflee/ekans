"""
S2 Cote d'Ivoire weekly cocoa port arrivals - data inaccessible.

Reuters publishes a weekly note from Abidjan reporting cumulative
season-to-date Ivorian cocoa port arrivals (Abidjan + San Pedro). The
weekly figures are reported in news articles only - there is no public
historical CSV or API and the Reuters paywall blocks archival scraping.

Alternative sources attempted:
- agroberichten.nl / DG Cafe-Cacao - Ivorian regulator only releases
  monthly cumulative tonnage in PDF press releases with inconsistent format.
- icco.org - International Cocoa Organization quarterly bulletin paywalled.
- afrik21.africa / cocoainitiative.org - editorial only, no time-series.

Without a structured weekly series the trigger (cumulative arrivals running
> 10% below prior season) cannot be evaluated systematically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "S2_civ_port_arrivals",
        "No public structured historical series for weekly Ivorian cocoa port arrivals; "
        "Reuters and ICCO data sit behind paywalls and DG Cafe-Cacao publishes only inconsistent monthly PDFs.",
        extra={
            "rule": "Long CC=F when cumulative season-to-date Ivorian port arrivals run > 10% below prior season's same week.",
            "mechanism": "Arrivals deficit -> West African supply shortfall -> tighter physical market -> CC=F rally.",
            "source_attempted": "Reuters Abidjan weekly note (paywall); DG Cafe-Cacao Cote d'Ivoire (PDF only, inconsistent); ICCO (paywall).",
        },
    )


if __name__ == "__main__":
    main()
