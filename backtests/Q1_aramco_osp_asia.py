"""
Q1 Aramco monthly Asia OSP -> Brent.

Aramco publishes Asia OSPs on the 5th of each month. There is no clean free CSV;
Reuters/Bloomberg headlines summarize differentials. Scraping Reuters mirror or
aramco.com requires an HTML scraper plus tag-level extraction that's brittle and
will degrade quickly.

We do not curate by hand for this signal (rule is "MoM raise > $1.50/bbl" - need
~150+ observations to be useful). Marking failed and citing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q1_aramco_osp_asia",
        "Aramco monthly Asia OSP differentials are not available as a free machine-readable feed. "
        "aramco.com publishes only the current month's PDF, and Reuters/Argus pricing summaries are paywalled. "
        "Scraping the past 10+ years of OSP press releases would require building a brittle PDF/HTML extractor; "
        "unjustified for a single signal.",
        extra={
            "source_attempted": ["https://www.aramco.com/en/news-media", "Reuters newswire archive"],
            "rule_intended": "Long Brent (BZ=F) 30 days when Aramco Asia OSP MoM raise > $1.50/bbl.",
            "mechanism": "Aramco OSP hike = strong Asian demand pull -> physical Dubai/Brent tightness.",
        },
    )


if __name__ == "__main__":
    main()
