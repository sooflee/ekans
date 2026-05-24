"""
Y3 Estimize top-analyst cluster signal.

Idea: Estimize tags each contributor with a track-record / "weight". When
multiple top-decile analysts (by historical accuracy) cluster on the
same side of the IBES consensus before earnings, follow their direction.

Why we mark this failed:
  - Requires analyst-level (not just consensus-level) Estimize history
    behind Nasdaq Data Link enterprise.
  - Requires the Estimize accuracy track-record metric, which is also
    paywalled.

Intended replication:
  1. For each earnings event identify top-decile analysts by trailing
     1-year hit rate / weighted MAE.
  2. Compute the share of top analysts above (below) IBES consensus.
  3. If ≥70% of top-decile analysts are above IBES → long at T-3, sell
     at T+1; if ≥70% below → short.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y3_estimize_top_analyst_cluster",
        ("Requires analyst-level Estimize estimate panel with the platform's "
         "internal accuracy track-record / weight per analyst. Both are "
         "now Nasdaq-Data-Link-enterprise gated. Free Kaggle dump (2012-2019) "
         "ends too early to give us a usable post-2019 OOS window."),
        extra={
            "rule": ("INTENDED: when ≥70% of top-decile Estimize analysts "
                     "(by trailing accuracy) lean the same direction vs "
                     "IBES consensus, long/short at T-3, exit T+1."),
            "mechanism": ("Skill-weighted consensus aggregation; experienced "
                          "buy-side & sell-side contributors carry private "
                          "info that has not yet hit IBES."),
            "source": ("Jame et al. (JAR 2016); Adebambo & Yan (RAS 2019); "
                       "Banerjee, Davis, Gondhi (2023 WP) on weighted-crowd "
                       "consensus."),
            "data_required": ("Estimize per-analyst estimate panel + Estimize "
                              "track-record / accuracy stat; IBES consensus."),
            "api_paywall": "Estimize via Nasdaq Data Link enterprise tier.",
        },
    )
    print("Y3 Estimize top-analyst cluster: marked failed (paywalled API).")


if __name__ == "__main__":
    main()
