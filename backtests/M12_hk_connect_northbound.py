"""
M12 HK Connect Northbound net buy.
Plan: pull daily Connect northbound net-buy series from HKEX.
Reality: HKEX's daily Connect CSV is served behind a JS-rendered SPA;
direct media/CSV URLs return 404/403 without scraping the React app.
No clean free public CSV dump is available.

Marking failed honestly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "M12_hk_connect_northbound",
        "HKEX daily Connect data only accessible via JS-rendered statistics pages; direct CSV/XLSX URLs return 404/403.",
        extra={
            "mechanism": "Mainland northbound flows >RMB 50B/10d reflect strong onshore PBOC-supported risk appetite; predictive of FXI 30d.",
            "source_attempted": "hkex.com.hk Mutual-Market/Stock-Connect statistics pages and media XLSX URLs.",
        },
    )


if __name__ == "__main__":
    main()
