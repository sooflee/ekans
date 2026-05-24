"""
Z9 NLRB Section 8(a) unfair-labor-practice complaint -- marked failed.

Why we mark this failed:
- NLRB's case-search interface (nlrb.gov/search/case) returns HTML
  result pages. The case detail pages name the "Charged Party" but
  the field is a free-text employer name, not a CIK / ticker.
- An NLRB charge is filed by an employee or union, not by NLRB
  itself; the regulatorily-meaningful event is when the Regional
  Director issues a *complaint* (i.e., decides the charge has merit).
  This sub-event is not exposed in the public case-search JSON --
  you have to read the docket entries inside each case folder.
- Mapping employer names to public-issuer tickers via fuzzy match is
  noisy (subsidiary names, franchisee splits, parent restatements).
- A free-tier, scriptable path from "Section 8(a) complaint issued by
  Regional Director" -> "public-issuer ticker" was not achievable.

Intended replication:
  1. Daily-poll the NLRB case search for case_type=CA (Section 8(a)).
  2. For each new case, scrape the docket to detect the
     "Complaint and Notice of Hearing" entry by the Regional Director.
  3. Fuzzy-map Charged Party employer name -> SEC EDGAR company name
     -> CIK -> ticker.
  4. Short the issuer T+1 for 60 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Z9_nlrb_complaint",
        ("NLRB case search exposes HTML results with the Charged Party as a "
         "free-text employer string. The regulatorily-relevant sub-event "
         "(Regional Director issues a Section 8(a) complaint) is inside the "
         "docket of each case folder, not in the search index. Combined "
         "with the noisy parent-vs-subsidiary fuzzy match to public-issuer "
         "tickers, there is no clean scriptable path to a defendant panel."),
        extra={
            "rule": ("INTENDED: for each Section 8(a) complaint issued by an "
                     "NLRB Regional Director, fuzzy-match Charged Party to a "
                     "public-issuer ticker; short the issuer T+1 for 60 "
                     "trading days."),
            "mechanism": ("A complaint (as distinct from the underlying "
                          "charge) is the Regional Director's determination "
                          "that the charge has merit; widely seen as a "
                          "credible signal of labor-litigation overhang."),
            "source": ("NLRB case search (HTML only); SEC EDGAR company name "
                       "search for fuzzy matching."),
            "data_obstacle": ("Charged Party is a free-text employer string "
                              "with subsidiary / franchisee ambiguity. The "
                              "complaint sub-event lives in case-docket "
                              "entries, not the search index."),
        },
    )
    print("Z9 NLRB complaint: marked failed (no clean defendant-ticker map).")


if __name__ == "__main__":
    main()
