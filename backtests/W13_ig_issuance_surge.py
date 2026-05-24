"""
W13 Investment-Grade Issuance Surge → short HYG, long LQD.

Mechanism: when monthly IG bond issuance is more than +1.5 sigma above
its 12-month rolling mean, the new supply weighs on IG spreads (LQD
under-performs) and a flight-to-quality dynamic widens HY spreads (HYG
under-performs even more). The HYG-LQD pair therefore loses; betting
short HY / long IG profits.

Data: SIFMA US Corporate Bonds Statistics page contains the monthly
issuance data, but the downloadable Excel link is gated behind a JS
button and not exposed in the rendered HTML (verified: HTML for
sifma.org/research/statistics/us-corporate-bonds-statistics/ has no
direct .xls/.xlsx href).

Fallback we attempted:
  - FRED corporate-debt issuance proxies (BUSLOANS is bank loans, not bond
    issuance; NCBDBIQ027S is quarterly liabilities-stock not monthly flow).
  - Nothing matches "monthly IG issuance" at clean frequency.

Without that monthly series the spec rule cannot be implemented cleanly.
We mark failed with a precise reproduction recipe.

Source: Greenwood-Hanson (RFS 2013, JF 2013) 'Issuer Quality and Corporate
        Bond Returns'; updated by Bordalo-Gennaioli-Shleifer (2024).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W13_ig_issuance_surge",
        ("SIFMA's monthly IG-issuance Excel is not exposed as a direct .xls "
         "link in the rendered statistics page; the download requires a "
         "JS-driven button click. FRED has no monthly IG-issuance series "
         "(BUSLOANS = bank loans, NCBDBIQ027S = quarterly liabilities stock). "
         "Without that monthly flow series the +1.5σ-vs-12m-mean rule cannot "
         "be implemented cleanly. Marked failed with the cite and reproduction "
         "recipe."),
        extra={
            "rule": ("INTENDED: monthly z-score of US IG corporate-bond issuance "
                      "vs 12m trailing mean and std. When z > +1.5, short HYG "
                      "and long LQD for the next 60 trading days. Pair PnL."),
            "universe": "HYG, LQD (long-short pair).",
            "source": ("Greenwood & Hanson (RFS 2013, JF 2013) 'Issuer Quality "
                        "and Corporate Bond Returns'. Monthly issuance data is "
                        "published by SIFMA US Corporate Bonds Statistics, but "
                        "the Excel link is dynamic."),
            "data_attempts": [
                "sifma.org/wp-content/uploads/.../US-Corporate-Bond-Statistics.xls → 404",
                "sifma.org/research/statistics/us-corporate-bonds-statistics/ → no static xls href",
                "FRED BUSLOANS → bank loans, not bond issuance",
                "FRED NCBDBIQ027S → quarterly liabilities stock, not monthly flow",
            ],
            "next_step_to_unblock": ("Either headless-browser SIFMA xls download, "
                                       "or pay for Bloomberg DCM Insight monthly "
                                       "issuance feed."),
        },
    )
    print("W13 IG issuance surge: marked failed (no monthly IG-issuance feed).")


if __name__ == "__main__":
    main()
