"""
AD-S3 Late-filer bunch fade.

Rule (spec): Chronic late-filer (>60-day avg lag — Tuberville, Fallon,
Crenshaw, Blake Moore) dumps >=10 backlog transactions on single PTR,
>=3 sector-concentrated → SHORT/fade sector ETF for 30-45 days.

This requires per-member filing-lag history aggregated across all PTRs,
which is not available in any free aggregator's JSON or CSV export. The
nearest free sources (HouseStockWatcher.com, SenateStockWatcher.com) post
individual PTR JSON but do NOT expose a clean per-member rolling lag
metric, and a backfill requires a months-long scrape of every member's
disclosure page.

Marking as FAILED per spec.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "AD-S3",
        "Requires per-member filing-lag history not in free aggregator "
        "format. HouseStockWatcher / SenateStockWatcher expose individual "
        "PTRs but no rolling per-member lag metric; backfilling would "
        "require months of scraping.",
        extra={
            "rule": "Chronic late-filer (>60d avg lag) dumps >=10 backlog "
                    "txns on single PTR, >=3 sector-concentrated -> short "
                    "sector ETF for 30-45 days.",
            "mechanism": "Stale trades + ethics-complaint news cycle adds "
                        "fade pressure to disclosed positions.",
            "source": "Spec: research/01ad_asymmetric_osint.md AD-S3.",
        },
    )
    print("AD-S3 marked FAILED — per-member filing-lag history not in free format.")


if __name__ == "__main__":
    main()
