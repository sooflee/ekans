"""
O2 FICC sponsored repo activity.

DATA STATUS: DTCC publishes the GCF Repo Index page (https://www.dtcc.com/charts/
dtcc-gcf-repo-index) as a JS dashboard; the static HTML returns HTTP 403 for
direct urllib requests and exposes no flat-file under /data or /downloads.
FICC Sponsored DVP volumes are released monthly by DTCC inside PDFs / blog posts;
there is no machine-readable time series.

OFR's short-term funding monitor and Fed Z.1 capture the FICC repo sleeve only at
quarterly granularity and tagged differently. A meaningful daily / weekly backtest
would require scraping the JS dashboard via Selenium or a paid data vendor.
Out of scope -> mark failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "O2_ficc_sponsored_repo",
        "DTCC GCF/FICC sponsored repo data is dynamic JS-rendered (HTTP 403 on direct fetch); no public CSV/XLSX time series. Scraping requires Selenium or paid feed.",
        extra={
            "rule_intended": "Use FICC Sponsored repo notional time series; pair with rates / risk assets.",
            "source": "DTCC GCF Repo Index (https://www.dtcc.com/charts/dtcc-gcf-repo-index)",
        },
    )
    print("O2 FICC sponsored repo: marked failed (dynamic JS dashboard, no free flat file).")


if __name__ == "__main__":
    main()
