"""
N3 AAR weekly rail traffic - intermodal.
AAR (Association of American Railroads) publishes weekly rail traffic data as PDFs
on aar.org/data-center/rail-traffic-data. Each PDF has a different layout; building
a robust historical scraper for ~15 years of weekly PDF tables is beyond this batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N3_aar_intermodal",
        "AAR weekly rail-traffic reports distributed only as per-week PDF press releases; no public CSV history.",
        extra={
            "rule_intended": "Long rails (UNP/CSX/NSC/CNI) when 4wk intermodal YoY turns positive after a contraction.",
            "source": "aar.org/data-center/rail-traffic-data",
            "universe": "UNP, CSX, NSC, CNI",
        },
    )
    print("N3 AAR intermodal: marked failed (per-week PDF scraping too heavy).")


if __name__ == "__main__":
    main()
