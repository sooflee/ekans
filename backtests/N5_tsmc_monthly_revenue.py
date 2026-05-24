"""
N5 TSMC monthly NT$ revenue press releases.
yfinance only carries annual income statement; TSMC monthly revenue is published as
press releases on tsmc.com (PDF / HTML) and aggregated by paid feeds. Curating ~180
monthly numbers across 15 years from disparate IR releases is heavier than this batch
allows. Mark failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N5_tsmc_monthly_revenue",
        "TSMC monthly NT$ revenue not available as a machine-readable free time series; yfinance carries only annual statements.",
        extra={
            "rule_intended": "Long {NVDA, AVGO, AMD, ASML} 10d when TSMC monthly YoY beats trailing-3m YoY avg by >+8pp.",
            "source": "TSMC IR monthly revenue press releases (https://investor.tsmc.com)",
            "universe": "NVDA, AVGO, AMD, ASML basket",
        },
    )
    print("N5 TSMC monthly revenue: marked failed (data unavailable as free time series).")


if __name__ == "__main__":
    main()
