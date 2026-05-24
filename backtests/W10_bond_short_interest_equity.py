"""
W10 Bond-ETF short interest spike → de-risk equity.

Original signal: when aggregate short interest in HY/IG bond ETFs (LQD, HYG,
JNK) spikes more than +1 sigma above a 24-month rolling mean (% of shares
outstanding), credit-quality dealers are positioning for spread widening.
Rule: halve SPY exposure for the following month.

Data availability check (May 2026):
  - FINRA cdn.finra.org/equity/regsho/monthly/ bulk download returns 403.
  - FINRA Open Data API returns 401 (auth required) for short-interest datasets.
  - Nasdaq's free short-interest API explicitly refuses ETF requests
    ("Short interest is only supported for Nasdaq Listed stocks").
  - Stockanalysis.com 404 on /etf/hyg/statistics.
  - WSJ short-interest page is gated by datadome captcha.

We cannot scrape historical bi-weekly bond-ETF short-interest free in this
session. Marked failed with the cite. A clean future implementation would:
  1. Pull FINRA bulk monthly short-interest text files (requires registered
     S3 credentials or screen-scraping the report search at
     https://reportcenter.finra.org/ which is JS-driven).
  2. For each settlement date, compute (shares_short / shares_outstanding) for
     HYG, LQD, JNK and a weighted average.
  3. Rolling 24-month z-score → de-risk SPY when z > +1.

Source: Bao, Pan, Wang (2024 working paper) 'Credit ETF Shorting and Equity
        Returns'.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W10_bond_short_interest_equity",
        ("Bond ETF (HYG/LQD/JNK) bi-weekly short interest is not freely "
         "downloadable: FINRA cdn returns 403, FINRA Open Data API requires "
         "auth, Nasdaq SI API refuses ETF symbols ('Nasdaq Listed stocks only'), "
         "and the report-center search is JS-driven. We mark this failed with "
         "the cite rather than fabricate a proxy."),
        extra={
            "rule": ("INTENDED: aggregate (short-interest / shares-out) for "
                      "LQD + HYG + JNK on each FINRA settlement date; 24m "
                      "rolling z-score; when z > +1, halve SPY exposure for 1 month."),
            "universe": "SPY (the asset being timed); HYG/LQD/JNK supply the signal.",
            "source": ("Bao, Pan, Wang (2024 wp) 'Credit ETF Shorting and "
                        "Equity Returns'; FINRA Reg SHO short interest."),
            "data_attempts": [
                "https://cdn.finra.org/equity/regsho/monthly/ → 403 AccessDenied",
                "https://api.finra.org/data/group/equity/name/monthlyShortInterest → 401",
                "https://api.nasdaq.com/api/quote/HYG/short-interest → 'ETF not supported'",
                "https://stockanalysis.com/etf/hyg/statistics/ → 404",
            ],
        },
    )
    print("W10 bond-ETF short interest: marked failed (no free data path).")


if __name__ == "__main__":
    main()
