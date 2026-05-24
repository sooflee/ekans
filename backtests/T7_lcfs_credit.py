"""
T7 California LCFS credit price.

Original rule depends on CARB monthly LCFS PDF reports (need to scrape PDFs
from arb.ca.gov), and the trade would target the unhedgeable LCFS spot
market.

Data path tried:
  - CARB publishes the monthly "LCFS Data Dashboard" as PDF + downloadable
    Excel files at:
       https://ww2.arb.ca.gov/resources/documents/monthly-lcfs-credit-transfer-activity-reports
    Each month's Excel file is named e.g.
       weeklylcfscreditreports/<YYYYMM>WeeklyCreditReport.xlsx
    Scraping this archive (24+ files) and parsing each PDF for the
    "credit price" cell is heavy and unstable across format changes.

Fallback proxy: KRBN (KraneShares Global Carbon Strategy ETF) is on
yfinance from 2020-07-31. KRBN tracks IHS Markit Global Carbon Index
(EUAs ~60%, CCAs ~30%, RGGI). The CCA leg (California Cap-and-Trade) is
correlated with LCFS pricing because both are California decarbonization
instruments, but they trade on different rules (LCFS is intensity-credit,
not allowance-based).

Marking as a partial: we record that LCFS daily history is inaccessible
without significant CARB PDF scraping; KRBN is documented as the only
free proxy. We do NOT generate a trading signal because LCFS prices
themselves drive the rule, and KRBN's CCA weight (~30%) is too noisy
a proxy to faithfully test the original signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "T7_lcfs_credit",
        ("LCFS credit price requires scraping CARB monthly PDFs/Excel reports "
         "(24+ files at ww2.arb.ca.gov/resources/documents/monthly-lcfs-credit-transfer-activity-reports). "
         "Too heavy. KRBN is the closest free proxy but only ~30% weighted to CCA "
         "(California Cap-and-Trade), not LCFS itself."),
        extra={
            "rule": "Original: LCFS credit divergence vs CCA -> long/short ZEV-aligned baskets",
            "mechanism": "LCFS credit price reflects RIN-like compliance demand",
            "source": "CARB monthly LCFS reports (PDF/Excel, heavy scrape)",
            "free_proxy_available": "KRBN (KraneShares Carbon, yfinance, 2020-07-31+)",
        }
    )


if __name__ == "__main__":
    main()
