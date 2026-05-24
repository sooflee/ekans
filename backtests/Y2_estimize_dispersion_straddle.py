"""
Y2 Estimize dispersion → long straddle.

Idea: Cross-analyst dispersion in the Estimize crowd-EPS consensus is a
forward-looking proxy for realized earnings-day volatility. When dispersion
is in the top decile relative to the stock's own history, buy a
near-ATM straddle 1-2 days before earnings and sell into the print.

Why we mark this failed:
  - Same data wall as Y1: per-event Estimize estimate panel is behind
    Nasdaq Data Link enterprise.
  - In addition needs single-name option chains (daily bid/ask) for ~2000
    earnings events / year. That is OptionMetrics-grade and not available
    via yfinance.

Intended replication:
  1. For each earnings event compute dispersion = stdev of Estimize EPS
     estimates / median estimate.
  2. Filter top decile cross-sectionally.
  3. Buy ATM straddle (call+put, same expiry, expiry ≥ T+5d) at close T-2;
     sell at close T+1.
  4. Equal-weight basket; vega-neutral construction optional.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y2_estimize_dispersion_straddle",
        ("Requires both Estimize crowd EPS-dispersion (Nasdaq Data Link "
         "enterprise) and single-name option chains around each earnings "
         "event (OptionMetrics or equivalent). Neither is available on a "
         "free public tier."),
        extra={
            "rule": ("INTENDED: long ATM straddle (T-2 → T+1) on stocks "
                     "with top-decile cross-analyst Estimize EPS "
                     "dispersion at the time of the print."),
            "mechanism": ("High crowd disagreement → high uncertainty → "
                          "realized vol on print > implied vol priced in "
                          "the straddle (Diether-Malloy-Scherbina logic in "
                          "options space)."),
            "source": ("Diether, Malloy, Scherbina (JF 2002); applied to "
                       "Estimize panel in Banerjee-Kremer 2019 working "
                       "paper; Jame et al. (JAR 2016)."),
            "data_required": ("Estimize per-analyst EPS estimates; "
                              "OptionMetrics IvyDB or single-name daily "
                              "option chain history; estimated 30-60k "
                              "earnings events with usable options."),
            "api_paywall": ("Estimize → Nasdaq Data Link enterprise; "
                            "options → OptionMetrics / WRDS."),
        },
    )
    print("Y2 Estimize dispersion straddle: marked failed (paywalled APIs).")


if __name__ == "__main__":
    main()
