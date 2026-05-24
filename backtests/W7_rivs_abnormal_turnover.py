"""
W7 Realized-IV Spread (RIVS) × Abnormal Turnover.

Original signal: cross-sectional RIVS (realized vol minus implied vol) combined
with abnormal share turnover predicts subsequent equity returns. Requires
daily option chains for thousands of stocks plus a per-stock turnover
abnormality model.

We mark this failed:
  - The IV component needs end-of-day option chains across the cross section
    (OptionMetrics or paid alternative).
  - The turnover component is easy from yfinance but useless without the IV side.

A degenerate SPY-only proxy (RV vs VIX, abnormal SPY volume) collapses to a
single-name timing model that is not the published cross-section result, so
we don't ship it as W7.

Source: Goyal-Saretto (JFE 2009) 'Cross-section of option returns and volatility'
        + Gao-Ritter (JFE 2010) abnormal turnover. The combined RIVS×turnover
        composite has been developed in 2023-24 working papers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W7_rivs_abnormal_turnover",
        ("Requires per-stock daily implied-vol surfaces (OptionMetrics or paid "
         "equivalent) across thousands of names. A SPY-only RV/IV + volume "
         "proxy collapses to a single-name timing test, which is not the "
         "published cross-sectional result. Honest fail rather than ship a "
         "weak proxy."),
        extra={
            "rule": ("INTENDED: rank stocks by (realised-vol − implied-vol) × "
                      "(abnormal turnover z-score). Long top decile, short "
                      "bottom decile. Monthly rebalance."),
            "universe": "Intended: Russell 3000 with liquid options.",
            "source": ("Goyal-Saretto (JFE 2009) for the RIVS side; Gao-Ritter "
                        "(JFE 2010) for abnormal turnover; the combined composite "
                        "is in 2023-24 working papers."),
            "infra_required": ("Daily option chain ETL or OptionMetrics; "
                                "stock-level abnormal-turnover model; CRSP-grade "
                                "universe."),
        },
    )
    print("W7 RIVS×abnormal-turnover: marked failed (data cost).")


if __name__ == "__main__":
    main()
