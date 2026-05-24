"""
Y10 Insider-unlock filter (team+VC > 70% allocation).

Idea: Token cliff-unlock pre-event short (Y7) but filtered to events
where ≥70% of the unlocked tranche is allocated to team / VC / private
investors (as opposed to community / ecosystem / airdrop). The
intuition is that mercenary insiders mark-to-market the unlock by
selling immediately, while community-incentive recipients hold.

Why we mark this failed:
  - Per-event allocation splits (team vs VC vs community %) require
    parsing each project's tokenomics PDF or relying on
    token.unlocks.app's structured breakdown, which is paid-tier-only.
  - For 5-10 events I could hand-curate splits, but the resulting
    sample would be too small and biased by which projects publish
    legible tokenomics (selection bias).
  - The honest answer per spec: mark_failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y10_insider_unlock_filter",
        ("Filtering token cliff-unlock events to those where >70% of the "
         "tranche goes to team/VC requires structured per-event allocation "
         "splits from token.unlocks.app paid tier or hand-curation from "
         "tokenomics PDFs. The hand-curated subset is too small and "
         "biased by which projects disclose legibly."),
        extra={
            "rule": ("INTENDED: Y7 cliff-unlock short, but only when the "
                     "tranche has >=70% team/VC/private-investor "
                     "allocation; expect larger and more reliable "
                     "negative pre-event drift than community-allocation "
                     "unlocks."),
            "mechanism": ("Team + VC recipients have mercenary holding "
                          "behavior — they sell on or shortly after vest "
                          "to lock dollar PnL. Community recipients "
                          "(airdrop, ecosystem grants) hold longer."),
            "source": ("token.unlocks.app event-level allocation tab; "
                       "Cong-Li-Wang (RFS 2021); Allen-Chen-Lo-Tao "
                       "(2024 WP) heterogeneity tests on allocation "
                       "type."),
            "data_required": ("Per-event tranche-allocation split "
                              "(team %, VC %, private %, community %, "
                              "ecosystem %); typically only available "
                              "on paid token.unlocks.app tier."),
        },
    )
    print("Y10 insider-unlock filter: marked failed (allocation metadata paywalled).")


if __name__ == "__main__":
    main()
