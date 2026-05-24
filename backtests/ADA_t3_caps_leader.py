"""
AD-T3 Named-foreign-leader caps → defense long / EM short pair.

Rule: Truth post mentions {Xi/Putin/Zelensky/Netanyahu/Kim/Khamenei/Maduro}
AND post is >=40% ALL CAPS or >=3 exclamations AND NOT containing
{deal, agreement, ceasefire, truce} → long ITA + short matched EM ETF.

Data availability check (2026-05-24):
- Same situation as AD-T2: trumpstruth.org has no public RSS, factba.se
  endpoints removed, thetrumparchive.com searchly index returns HTTP 401
  on external requests.
- Without a parseable post archive with full text + timestamps, we cannot
  enumerate the historical event set required to compute the strategy's
  return distribution.

Marking as FAILED.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "AD-T3",
        "Cannot enumerate Trump posts containing named-leader + caps/!! "
        "pattern: no parseable bulk Trump-post archive is currently free. "
        "trumpstruth.org has no public feed, factba.se JSON endpoints "
        "removed, thetrumparchive.com searchly backend returns HTTP 401 "
        "(verified 2026-05-24).",
        extra={
            "rule": "Trump post mentions {Xi/Putin/Zelensky/Netanyahu/Kim/"
                    "Khamenei/Maduro} AND >=40% caps OR >=3 '!' AND NOT in "
                    "{deal/agreement/ceasefire/truce} -> long ITA / short "
                    "matched EM ETF for 3 days.",
            "mechanism": "Posts about named adversaries with hostile tone "
                        "presage escalation; markets re-price defense vs "
                        "exposed-country risk.",
            "source": "Spec: research/01ad_asymmetric_osint.md AD-T3.",
        },
    )
    print("AD-T3 marked FAILED — Trump-post archive not accessible.")


if __name__ == "__main__":
    main()
