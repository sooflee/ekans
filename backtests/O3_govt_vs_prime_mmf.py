"""
O3 Government vs Prime money-market fund weekly asset growth rotation.

Intended rule: When Govt MMF weekly growth > Prime MMF weekly growth for 3 consecutive
weeks, long XLU / short XLF as a flight-to-quality pair for 8 weeks.

DATA STATUS: ICI publishes the weekly MMF Total Net Assets breakdown only as XLS
on its public website (https://www.ici.org/research/stats/mmf), and on probing only
the *current-year* file (mm_summary_data_2025.xls) is reachable; years 2010-2024
404. FRED carries quarterly aggregate MMF assets (MMMFFAQ027S) but no Govt-vs-Prime
weekly split. OFR's MMF Monitor renders client-side from a JS dashboard; no public
flat-file. A meaningful multi-year backtest of the rotation would require scraping
archived ICI XLS via the Wayback Machine (~750 weekly snapshots) or paid
data feeds. That is out of scope for this batch -> mark failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "O3_govt_vs_prime_mmf",
        "ICI mm_summary_data XLS publishes only current year on the public site; FRED has no Govt-vs-Prime weekly split; OFR MMF Monitor is JS-rendered. Multi-year backtest infeasible without paid feed or Wayback scrape.",
        extra={
            "rule_intended": "When Govt MMF weekly growth > Prime MMF weekly growth for 3 consecutive weeks, long XLU / short XLF for 8 weeks.",
            "mechanism": "Flight-to-quality rotation among MMF investors -> defensive equity bid (XLU) vs financials drag.",
            "source": "ICI Money Market Funds weekly XLS (https://www.ici.org/research/stats/mmf)",
            "universe": "XLU/XLF pair",
        },
    )
    print("O3 govt-vs-prime MMF: marked failed (only current-year ICI XLS exposed publicly).")


if __name__ == "__main__":
    main()
