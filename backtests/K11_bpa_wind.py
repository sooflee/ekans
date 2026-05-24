"""
K11 BPA wind forecast vs actual — long FAN on 3-consecutive-day underperformance.

Status: data not available for a multi-year backtest.

BPA transmission.bpa.gov serves only a rolling 7-day window of wind
generation vs base-schedule (file: twndbspt.txt). The historical archive
page (Wind_Forecast/Archives.aspx) does not expose a downloadable CSV/XLS
of past forecasts. Without years of forecast vs actual history we cannot
construct the "3-consecutive-day forecast underperforms by >25%" signal,
so we mark this signal as failed rather than synthesize from a 7-day
snapshot.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed("K11_bpa_wind",
                "BPA transmission.bpa.gov only serves last-7-days wind/forecast in "
                "twndbspt.txt; the Wind_Forecast Archives.aspx page does not link "
                "downloadable historical CSV/XLS. Without years of forecast-vs-actual "
                "history a backtest of '3 consecutive days underperform by >25%' is "
                "not constructible from public BPA endpoints.",
                extra={"source_checked": "transmission.bpa.gov twndbspt.txt + Wind_Forecast/Archives.aspx"})


if __name__ == "__main__":
    main()
