"""
L12 USDA corn Good/Excellent — requires NASS Quick Stats API key.

USDA NASS Quick Stats API at quickstats.nass.usda.gov/api requires a free
registered API key (key parameter is mandatory; unauthenticated requests
return {"error":["unauthorized"]}). The Crop Progress PDFs that contain
the weekly Good/Excellent corn condition % during July pollination would
need either (a) a registered key or (b) PDF scraping over the entire
2010-2024 archive.

FRED carries only the corn cash price (WPU012203) and not the
condition rating. The spec explicitly says to "manually curate" -- not a
viable automated backtest path.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L12_corn_pollination",
        ("USDA NASS Quick Stats requires an API key (free but registration "
         "gated); Crop Progress PDFs over 2010-2024 require manual curation "
         "per spec. No FRED proxy for corn G/E condition exists."),
        extra={
            "rule": ("July corn G/E condition WoW drop > 5 ppt -> long CORN ETF 4w "
                     "(intended rule; needs NASS API key or manual PDF curation)."),
            "mechanism": "Pollination-window deterioration in corn condition tightens forward supply.",
            "source_attempted": "https://quickstats.nass.usda.gov/api/api_GET/ (requires key).",
        },
    )


if __name__ == "__main__":
    main()
