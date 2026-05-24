"""
Y6 SPAC second-extension long-vol.

Idea: When a SPAC seeks a second deadline extension (i.e., already
exercised one extension and now asking shareholders again), it signals
either (a) imminent rushed bad-deal pre-announcement or (b) liquidation.
Either is volatility-positive. The capped downside is roughly the trust
NAV (~$10.05) since holders can redeem; the upside is a deal pop.

Why we mark this failed:
  - Identifying "second extension" events at scale requires programmatic
    EDGAR 8-K full-text search ("second extension", "extend completion
    period", "extension proposal") across thousands of SPAC tickers,
    then dedup/curation. That's the heavy ETL the task allowed us to
    mark_failed against if too sparse.
  - For a hand-curated subset (~5-8 events) the sample is statistically
    meaningless and biased by survivorship of memorable cases.
  - Even when we know an event date, SPAC pre-announcement common-stock
    options are typically not exchange-listed; this is a synthetic
    long-vol that needs OTC quotes.

Intended replication:
  1. EDGAR EDGAR full-text "form-type=8-K" + "second extension"
     between 2020-08 and 2023-12.
  2. Filter to SPACs (look up sponsor → trust structure).
  3. Buy SPAC common at ~$10.00-10.05 (trust NAV floor) before second
     extension vote; sell on (a) deal-announcement pop or (b) just
     before liquidation distribution.
  4. Hedge: pair with redemption right (effectively a put at NAV).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y6_spac_second_extension",
        ("Second-extension events require EDGAR 8-K full-text search "
         "across thousands of SPAC tickers + manual SPAC-status curation. "
         "A hand-picked subset is too small (n<10) for any meaningful "
         "t-stat. SPAC pre-announcement options are not exchange-listed, "
         "so the 'long-vol at capped downside' construction needs OTC "
         "quotes we don't have access to."),
        extra={
            "rule": ("INTENDED: buy SPAC common at NAV floor ~$10.05 "
                     "ahead of a *second* shareholder extension vote; "
                     "exit on deal announcement pop or pre-liquidation "
                     "distribution."),
            "mechanism": ("Trust redemption right caps downside near NAV; "
                          "second extension predicts a binary outcome "
                          "(rushed deal vs liquidation), both vol-positive."),
            "source": ("Klausner, Ohlrogge, Ruan (NBER 2022); SPACInsider "
                       "extension trackers; Gahng, Ritter, Zhang (RFS 2023) "
                       "discuss extension dynamics."),
            "data_required": ("EDGAR 8-K full-text search engine + SPAC "
                              "sponsor registry; per-SPAC trust-NAV time "
                              "series; OTC options for synthetic long-vol "
                              "expression."),
            "sample_size_problem": ("Hand-curated subset n<10 has no "
                                    "statistical power; better to ship "
                                    "as fail than as noise."),
        },
    )
    print("Y6 SPAC 2nd-extension: marked failed (data sparsity).")


if __name__ == "__main__":
    main()
