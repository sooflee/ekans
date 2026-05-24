"""
AD-T2 Trump burst-rate intensity → VIX long.

Rule (per research note): if Trump posts ≥25 times in a rolling 60-min window
between 6pm-7am ET → buy VIX next 9:30 ET open; exit T+3 close.

Data availability check (2026-05-24):
- trumpstruth.org: SPA with no public RSS / JSON export (front-end blocked).
- factba.se: '/json/trumptweets/' and similar endpoints return 404.
- thetrumparchive.com: backed by searchly thorin-us-east-1; the JS bundle
  exposes app 'trump_tweets' and a public-key, but the searchly index now
  rejects all external requests with 'Unauthorized access' (HTTP 401),
  including the documented reactivesearch.v3 endpoint with proper Referer
  / Origin headers.
- No bulk timestamped Trump Truth archive is currently parseable for free
  in a way that lets us re-derive rolling 60-min post counts.

Marking as FAILED — data infrastructure required for this strategy is no
longer accessible on the public web.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "AD-T2",
        "Cannot derive rolling 60-min Trump post counts: trumpstruth.org "
        "no longer exposes RSS, factba.se JSON endpoints removed, and the "
        "thetrumparchive.com searchly backend returns HTTP 401 on every "
        "external query (verified 2026-05-24). Strategy needs a real-time "
        "scraper run alongside live posts, not a back-tested archive.",
        extra={
            "rule": "Trump posts >=25 in rolling 60-min window between "
                    "6pm-7am ET -> long VIX next open, exit T+3 close.",
            "mechanism": "Manic posting bursts empirically precede policy "
                        "reversals, firings, escalations.",
            "source": "Spec: research/01ad_asymmetric_osint.md AD-T2.",
            "attempted_sources": [
                "trumpstruth.org RSS - 404",
                "factba.se JSON endpoints - 404",
                "thetrumparchive.com searchly _search / _reactivesearch.v3 - HTTP 401",
            ],
        },
    )
    print("AD-T2 marked FAILED — Trump-post timestamped archive not accessible.")


if __name__ == "__main__":
    main()
