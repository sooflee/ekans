"""
AA2 ARKK front-run.
ARK publishes daily trade notifications via email/CathiesARK.com.
Historical archive of daily ARKK trades 2020-2024 with per-stock $-amounts is
not freely scrapable in this pass (CathiesARK requires bs4 + auth-free crawl
with rate-limiting and per-day pages × ~5 funds × ~10 names = ~50k rows).

We mark this signal as failed-by-data-availability rather than fabricate it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "AA2_arkk_front_run",
        "ARK daily trade history requires curated multi-year scrape (CathiesARK / ARK CSV archive) — out of scope for this pass.",
        extra={
            "source": "CathiesARK.com / ARK Invest daily trade emails",
            "universe": "ARKK constituents",
            "note": "Live implementation would subscribe to ARK trade emails and trade T+0 close → T+1 close on names with >0.5% AUM purchases.",
        },
    )
    print("AA2 ARKK front-run: marked failed (data not available in this pass).")


if __name__ == "__main__":
    main()
