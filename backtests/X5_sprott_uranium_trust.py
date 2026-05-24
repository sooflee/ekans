"""
X5 Sprott Physical Uranium Trust premium/discount to NAV.

Original spec: Use Sprott-published weekly NAV vs SRUUF/U-UN.TO market price.
When discount > 5%, long for mean-reversion; when premium > 10%, short.

Approach: NAV is published at sprott.com weekly (proprietary scrape, no public
free API). Without curated NAV history, we can attempt a *proxy* using the
trust's spot U3O8 price content (the trust holds physical pounds of U3O8 only).

Verified proxies:
- URNM: Sprott Junior Uranium Miners ETF
- URA: Global X Uranium ETF
- SRUUF: Sprott Physical Uranium Trust (US OTC)
- U-UN.TO: TSX listing
None expose NAV history without authenticated/scraped data.

Decision: mark_failed with full rationale rather than fabricate a NAV series.
The honest substitute (price-ratio mean reversion) was attempted but yields a
spurious signal because both SRUUF and uranium spot are largely driven by the
same factor — a "discount" proxy would mostly capture trading noise, not real NAV gap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    sid = "X5_sprott_uranium_trust"
    return mark_failed(
        sid,
        "Sprott Physical Uranium Trust weekly NAV is published only on sprott.com "
        "via dynamic JS dashboard; no public free API. Without authenticated NAV "
        "history we cannot compute genuine premium/discount. Free-data proxies "
        "(SRUUF vs URA/URNM ratio) capture covariance not NAV gap and would be "
        "misleading. Marking failed per protocol.",
        extra={
            "status": "fail",
            "rule": "(intended) Long SRUUF when discount to Sprott NAV > 5%; short when premium > 10%.",
            "mechanism": "Closed-end-like vehicle premium/discount mean-reversion.",
            "source_required": "sprott.com/investment-strategies/physical-commodity-funds/uranium daily/weekly NAV.",
            "blocker": "No public free API; weekly NAV is behind JS dashboard, requires curated scrape.",
            "universe": ["SRUUF", "U-UN.TO"],
        },
    )


if __name__ == "__main__":
    main()
