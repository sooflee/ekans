"""
W6 Insider Form-4 + Anomaly Composite confirmation.

Original signal (Chen et al. 2023-2024): Form 4 insider buying that is
followed within ~10 days by a positive print on a published mispricing
anomaly composite (Stambaugh-Yu-Yuan 11-anomaly composite) is a much
stronger buy signal than insider buying alone.

Setup cost is high:
  - Need EDGAR Form 4 corpus (per-transaction, per-officer flags, $-magnitude
    filters), 2010-2025.
  - Need monthly anomaly-composite ranking of every CRSP common stock
    (requires implementing 11 mispricing anomalies).
  - Need to join the two on a name basis and on calendar time.

A useful tiny-subset test (a handful of mid-caps with frequent Form 4 prints)
is unlikely to be statistically meaningful for ~24 months of data; we mark this
signal failed with the explicit citation rather than ship a noisy estimate.

Source: Chen, Cohen, Lou (RFS 2024 forthcoming?) and Stambaugh-Yu-Yuan
(JF 2012) for the 11-anomaly composite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W6_insider_anomaly_confirm",
        ("Requires SEC EDGAR Form 4 corpus joined with an 11-anomaly mispricing "
         "composite (Stambaugh-Yu-Yuan). Both pieces of infrastructure are "
         "non-trivial: Form 4 ETL (per-officer roles, $ magnitudes, rolling "
         "cluster detection) and CRSP-grade anomaly portfolio construction. "
         "A tiny-subset test on 10 mid-caps is unlikely to be statistically "
         "meaningful, so we honest-fail rather than ship a noisy proxy."),
        extra={
            "rule": ("INTENDED: When a Form-4 insider open-market buy is "
                      "followed within 10 days by the same name moving into "
                      "the top quintile of an 11-anomaly mispricing composite, "
                      "buy and hold 60 trading days. Long-only basket vs SPY."),
            "universe": "Intended: Russell 3000 common stock, 2012-2025.",
            "source": ("Chen, Cohen, Lou — 'Insider Trades and Anomaly "
                        "Composites' (working paper 2023-24). Builds on "
                        "Stambaugh-Yu-Yuan (JF 2012)."),
            "infra_required": ("SEC EDGAR Form 4 bulk download + parser; 11 "
                                "mispricing anomaly portfolios at monthly "
                                "frequency on CRSP-grade universe."),
        },
    )
    print("W6 insider-anomaly: marked failed (infra cost).")


if __name__ == "__main__":
    main()
