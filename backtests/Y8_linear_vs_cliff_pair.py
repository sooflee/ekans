"""
Y8 Linear-unlock vs cliff-unlock paired short.

Idea: Within the same sector (e.g., L2s), tokens with cliff-unlock
schedules should underperform those with linear-vest schedules in the
window around insider unlock dates, because cliffs concentrate selling
pressure while linear-vests are continuously digested.

Construction (intended):
  - For each Y7 cliff event, pair with a same-sector token whose
    unlock schedule is linear-vest (no cliff in window).
  - Long the linear-vest token, short the cliff-unlock token, hold
    T-7 → T+0.
  - Equal-weight basket of paired trades.

Why we mark this failed:
  - Requires structured unlock-schedule metadata per token
    (linear vs cliff, daily emission rate) which is only available on
    token.unlocks.app / cryptorank.io paid tiers or via heavy scraping
    of project tokenomics docs.
  - Mis-pairing sector noise (L2 vs L1 vs DeFi) dominates the signal.
    Without a defensible same-sector pairing matrix we'd be reporting
    arbitrary crypto pair returns.
  - Honest call: mark_failed rather than ship a hand-paired
    over-fit result.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y8_linear_vs_cliff_pair",
        ("Requires structured per-token unlock-schedule metadata "
         "(linear vs cliff, emission rate) plus a defensible same-sector "
         "pairing matrix. Both the schedule data and the sector "
         "classification require token.unlocks.app / cryptorank.io paid "
         "tiers or hand-scraped tokenomics docs. Hand-pairing across "
         "sectors with n<15 is overfit by construction."),
        extra={
            "rule": ("INTENDED: long linear-vest token / short cliff-unlock "
                     "token in the same sector around cliff date, T-7 to "
                     "T+0, equal-weight basket of pair trades."),
            "mechanism": ("Cliff concentrates a discrete supply shock; "
                          "linear-vest absorbs the same notional gradually "
                          "and the market has already priced its drag."),
            "source": ("token.unlocks.app and cryptorank.io tokenomics "
                       "tables; Allen-Chen-Lo-Tao (2024 WP); Cong-Li-Wang "
                       "(RFS 2021) on token-supply dynamics."),
            "data_required": ("Per-token (a) cliff vs linear schedule "
                              "metadata and (b) same-sector classification "
                              "to build defensible pairs."),
        },
    )
    print("Y8 linear-vs-cliff pair: marked failed (data + pairing complexity).")


if __name__ == "__main__":
    main()
