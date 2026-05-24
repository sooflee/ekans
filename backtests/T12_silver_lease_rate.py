"""
T12 Silver lease rate -> SI=F.

Rule: 1-month silver lease rate > 4% for 3 days -> long SI=F 2 months.

Data path tried:
  - LBMA stopped publishing public daily SIFO (silver forward offered rates)
    in 2015. The lease rate = LIBOR - SIFO, so post-2015 data is gone.
  - Kitco's historical silver lease page (kitco.com/leasing/silverlease.html)
    now returns 404. The /charts/historicalsilver.html page returns HTML
    but has no lease-rate data (only price chart).
  - BullionVault's silver lease chart page returns 404.

The few public historical silver lease snapshots that exist (e.g., on
trading-economics or seeking-alpha archive pages) are not date-indexed
time series we can backtest. Mark failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "T12_silver_lease_rate",
        ("LBMA stopped publishing daily SIFO in 2015; Kitco/BullionVault "
         "lease-rate pages now 404. No free daily silver lease rate time "
         "series is available."),
        extra={
            "rule": "1m silver lease > 4% for 3d -> long SI=F 2 months",
            "mechanism": "Lease spike = physical metal scarcity -> price upside",
            "source": ("LBMA SIFO (discontinued 2015), Kitco/BullionVault "
                       "(now 404)"),
        }
    )


if __name__ == "__main__":
    main()
