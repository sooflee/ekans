"""
Q8 Brazil sugar - UNICA Center-South crush.

UNICA bi-weekly crush data is hosted at observatoriodacana.com.br with HTML and PDFs
requiring scraping; the site has rate limiting and dynamic content. USDA WASDE sugar
stocks-to-use ratio is monthly and available via USDA PSD but requires CSV pulls
plus alignment - non-trivial.

Marking failed; would require dedicated UNICA scraper or USDA PSD ingestion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q8_brazil_sugar_crush",
        "UNICA Center-South bi-weekly crush data requires scraping observatoriodacana.com.br "
        "(dynamic content, rate-limited). USDA WASDE proxy needs separate PSD ingestion. "
        "Out of scope for a single signal.",
        extra={
            "source_attempted": [
                "https://observatoriodacana.com.br",
                "https://apps.fas.usda.gov/psdonline",
            ],
            "rule_intended": "When CS Brazil crush week-on-week drop > 15%, long SB=F or CANE for 30d.",
            "mechanism": "Bad crush -> tighter sugar; Brazil is the marginal exporter, so misses ripple to NY11.",
        },
    )


if __name__ == "__main__":
    main()
